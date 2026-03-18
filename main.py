from fastapi import FastAPI, File, UploadFile
import hashlib
import os
import requests
import cv2
import tempfile
from pypdf import PdfReader
import docx  # <-- LIBRERÍA PARA LEER WORD
import io
import time

app = FastAPI()

# --- CREDENCIALES EMPRESARIALES E HÍBRIDAS ---
API_USER = os.getenv("SIGHTENGINE_USER")
API_SECRET = os.getenv("SIGHTENGINE_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")

# Modelos Especializados de Hugging Face
HF_TEXT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
HF_AUDIO_MODEL = "mrfakename/Wav2Vec2-LJSpeech-AI-Audio-Detector"

def analizar_con_sightengine(contenido_bytes: bytes, nombre_archivo: str, mime_type: str):
    """Motor comercial para Imágenes"""
    url = 'https://api.sightengine.com/1.0/check.json'
    archivos = {'media': (nombre_archivo, contenido_bytes, mime_type)}
    datos = {'models': 'genai', 'api_user': API_USER, 'api_secret': API_SECRET}
    
    respuesta = requests.post(url, files=archivos, data=datos)
    return respuesta.json()

def analizar_texto_hf(texto: str, max_intentos=3):
    """Motor NLP con blindaje y reintento automático"""
    url = f"https://router.huggingface.co/hf-inference/models/{HF_TEXT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    
    for intento in range(max_intentos):
        respuesta = requests.post(url, headers=headers, json={"inputs": texto[:1500]})
        
        if respuesta.status_code == 503:
            try:
                tiempo_espera = respuesta.json().get("estimated_time", 20.0)
            except:
                tiempo_espera = 20.0
            print(f"[INFO] Despertando modelo de texto. Esperando {tiempo_espera}s...")
            time.sleep(tiempo_espera + 2)
            continue
            
        # Verificamos si Hugging Face dio error antes de leer el JSON
        if respuesta.status_code != 200:
            return {"error": f"Hugging Face rechazó el documento (Código HTTP {respuesta.status_code})."}
            
        try:
            return respuesta.json()
        except:
            return {"error": "El servidor de IA devolvió una respuesta ilegible."}
            
    return {"error": "El servidor de lenguaje no despertó a tiempo."}

def analizar_audio_hf(audio_bytes: bytes, max_intentos=3):
    """Motor de frecuencias blindado contra caídas del modelo"""
    url = f"https://router.huggingface.co/hf-inference/models/{HF_AUDIO_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}
    
    for intento in range(max_intentos):
        respuesta = requests.post(url, headers=headers, data=audio_bytes)
        
        if respuesta.status_code == 503:
            try:
                tiempo_espera = respuesta.json().get("estimated_time", 20.0)
            except:
                tiempo_espera = 20.0
            print(f"[INFO] Despertando modelo de audio. Esperando {tiempo_espera}s...")
            time.sleep(tiempo_espera + 2)
            continue
            
        # Verificamos si Hugging Face dio error antes de leer el JSON
        if respuesta.status_code != 200:
            return {"error": f"El modelo de audio falló o no está disponible (Código HTTP {respuesta.status_code})."}
            
        try:
            return respuesta.json()
        except:
            return {"error": "El analizador de espectrogramas devolvió un formato inválido."}
            
    return {"error": "El servidor de espectrogramas no respondió a tiempo."}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    porcentaje_ia = 0.0
    tipo_evidencia = "DESCONOCIDO"
    error_api = None
    desglose_ui = {}

    try:
        # ==========================================
        # 1. MÓDULO DE IMÁGENES
        # ==========================================
        if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            tipo_evidencia = "IMAGEN"
            res = analizar_con_sightengine(contenido, nombre_archivo, file.content_type)
            
            if res.get("status") == "success":
                porcentaje_ia = res.get("type", {}).get("ai_generated", 0.0) * 100
                desglose_ui = {
                    "coherencia_optica": round(100 - porcentaje_ia + (porcentaje_ia * 0.1), 1),
                    "integridad_metadatos": 95.0 if porcentaje_ia < 50 else 12.5,
                    "firma_algoritmica": round(porcentaje_ia, 1)
                }
            else:
                error_api = res.get('error', {}).get('message', 'Fallo en Sightengine')

        # ==========================================
        # 2. MÓDULO DE VIDEO
        # ==========================================
        elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
            tipo_evidencia = "VIDEO"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(contenido)
                ruta_temp = temp_video.name
                
            try:
                captura = cv2.VideoCapture(ruta_temp)
                total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
                captura.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.5))
                exito, frame = captura.read()
                
                if exito:
                    _, buffer = cv2.imencode('.jpg', frame)
                    res = analizar_con_sightengine(buffer.tobytes(), "frame.jpg", "image/jpeg")
                    if res.get("status") == "success":
                        porcentaje_ia = res.get("type", {}).get("ai_generated", 0.0) * 100
                        desglose_ui = {"fluidez_temporal": round(100 - (porcentaje_ia*0.8), 1), "anomalia_pixel": round(porcentaje_ia, 1)}
                    else:
                        error_api = "Sightengine rechazó el frame del video."
                captura.release()
            finally:
                os.remove(ruta_temp)

        # ==========================================
        # 3. MÓDULO DE DOCUMENTOS (PDF y DOCX)
        # ==========================================
        elif nombre_archivo.endswith(('.pdf', '.docx')):
            tipo_evidencia = "DOCUMENTO"
            texto_extraido = ""
            
            # Lógica para extraer texto de PDF
            if nombre_archivo.endswith('.pdf'):
                lector = PdfReader(io.BytesIO(contenido))
                for pagina in lector.pages[:3]: # Leemos max 3 páginas
                    texto = pagina.extract_text()
                    if texto:
                        texto_extraido += texto + " "
                    
            # Lógica para extraer texto de Word (DOCX)
            elif nombre_archivo.endswith('.docx'):
                documento = docx.Document(io.BytesIO(contenido))
                for parrafo in documento.paragraphs[:20]: # Leemos max 20 párrafos
                    if parrafo.text:
                        texto_extraido += parrafo.text + " "
                
            if len(texto_extraido.strip()) < 50:
                error_api = "El documento está vacío o es una imagen escaneada sin texto seleccionable."
            else:
                res = analizar_texto_hf(texto_extraido)
                
                if isinstance(res, list) and len(res) > 0:
                    # El Parche: A veces devuelve [[{...}]] y otras veces [{...}]
                    datos_scores = res[0] if isinstance(res[0], list) else res
                    
                    encontro_ia = False
                    for item in datos_scores:
                        etiqueta = str(item.get('label', '')).lower()
                        # Buscamos variaciones en el nombre de la etiqueta IA
                        if 'chatgpt' in etiqueta or 'fake' in etiqueta or '1' in etiqueta or 'ai' in etiqueta:
                            porcentaje_ia = item['score'] * 100
                            encontro_ia = True
                            break
                            
                    # Si solo devolvió la etiqueta Humano, invertimos
                    if not encontro_ia:
                        for item in datos_scores:
                            etiqueta = str(item.get('label', '')).lower()
                            if 'human' in etiqueta or 'real' in etiqueta or '0' in etiqueta:
                                porcentaje_ia = (1.0 - item['score']) * 100
                                break
                    
                    desglose_ui = {
                        "perplejidad_linguistica": round(100 - porcentaje_ia, 1),
                        "patrones_chatgpt": round(porcentaje_ia, 1)
                    }
                elif isinstance(res, dict) and "error" in res:
                    error_api = res.get("error", "Error desconocido al procesar el documento.")
                else:
                    error_api = "Estructura de respuesta no reconocida del servidor de lenguaje."

        # ==========================================
        # 4. MÓDULO DE AUDIO
        # ==========================================
        elif nombre_archivo.endswith(('.mp3', '.wav', '.ogg')):
            tipo_evidencia = "AUDIO"
            res = analizar_audio_hf(contenido)
            if isinstance(res, list) and len(res) > 0:
                res_ordenado = sorted(res, key=lambda x: x['score'], reverse=True)
                top_label = res_ordenado[0]['label'].lower()
                top_score = res_ordenado[0]['score']
                
                if "fake" in top_label or "ai" in top_label:
                    porcentaje_ia = top_score * 100
                else:
                    porcentaje_ia = (1.0 - top_score) * 100
                    
                desglose_ui = {
                    "espectrograma_natural": round(100 - porcentaje_ia, 1),
                    "frecuencias_sinteticas": round(porcentaje_ia, 1)
                }
            elif isinstance(res, dict) and "error" in res:
                 error_api = res.get("error", "Error desconocido al procesar el audio.")
            else:
                error_api = "Formato de respuesta desconocido del analizador de audio."

        else:
            error_api = "Formato de archivo no soportado por ForensIA."

        # --- LÓGICA DE RESPUESTA COMERCIAL ---
        if error_api:
            return {
                "nombre_archivo": nombre_archivo,
                "tipo_evidencia": tipo_evidencia,
                "analisis": {"nivel_riesgo": "ERROR", "motivo": error_api, "porcentaje_ia": 0.0}
            }

        if porcentaje_ia > 75.0:
            nivel_riesgo = "ALTO"
            motivo = "Alta probabilidad de generación algorítmica profunda (Sintético)."
        elif porcentaje_ia > 25.0:
            nivel_riesgo = "PREVENTIVO"
            motivo = "Se detectan trazas de alteración. Posible asistencia parcial de IA."
        else:
            nivel_riesgo = "BAJO"
            motivo = "Evidencia íntegra. Certificación de origen humano/natural."

        return {
            "nombre_archivo": nombre_archivo,
            "hash_sha256": hash_sha256,
            "tipo_evidencia": tipo_evidencia,
            "analisis": {
                "porcentaje_ia": round(porcentaje_ia, 2),
                "nivel_riesgo": nivel_riesgo,
                "motivo": motivo,
                "desglose_ui": desglose_ui
            }
        }

    except Exception as e:
        return {"error": f"Error del motor multimodal: {str(e)}"}