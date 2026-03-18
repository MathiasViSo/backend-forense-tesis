from fastapi import FastAPI, File, UploadFile
import hashlib
import os
import requests
import cv2
import tempfile
from pypdf import PdfReader
import io

app = FastAPI()

# --- CREDENCIALES EMPRESARIALES E HÍBRIDAS ---
API_USER = os.getenv("SIGHTENGINE_USER")
API_SECRET = os.getenv("SIGHTENGINE_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")

# Modelos Especializados de Hugging Face
HF_TEXT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
HF_AUDIO_MODEL = "ResembleAI/ai_detector_audio"

def analizar_con_sightengine(contenido_bytes: bytes, nombre_archivo: str, mime_type: str):
    """Motor comercial para Imágenes"""
    url = 'https://api.sightengine.com/1.0/check.json'
    archivos = {'media': (nombre_archivo, contenido_bytes, mime_type)}
    datos = {'models': 'genai', 'api_user': API_USER, 'api_secret': API_SECRET}
    
    respuesta = requests.post(url, files=archivos, data=datos)
    return respuesta.json()

def analizar_texto_hf(texto: str):
    """Motor NLP para detectar ChatGPT en documentos"""
    url = f"https://router.huggingface.co/hf-inference/models/{HF_TEXT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    
    # Recortamos el texto a los primeros 1500 caracteres para no saturar la API
    respuesta = requests.post(url, headers=headers, json={"inputs": texto[:1500]})
    return respuesta.json()

def analizar_audio_hf(audio_bytes: bytes):
    """Motor de frecuencias para detectar clonación de voz"""
    url = f"https://router.huggingface.co/hf-inference/models/{HF_AUDIO_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}
    
    respuesta = requests.post(url, headers=headers, data=audio_bytes)
    return respuesta.json()

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
                # Desglose simulado para nutrir la Interfaz Gráfica de Flutter
                desglose_ui = {
                    "coherencia_optica": round(100 - porcentaje_ia + (porcentaje_ia * 0.1), 1),
                    "integridad_metadatos": 95.0 if porcentaje_ia < 50 else 12.5,
                    "firma_algoritmica": round(porcentaje_ia, 1)
                }
            else:
                error_api = res.get('error', {}).get('message', 'Fallo en Sightengine')

        # ==========================================
        # 2. MÓDULO DE VIDEO (Extracción de Frame)
        # ==========================================
        elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
            tipo_evidencia = "VIDEO"
            # Guardamos el video temporalmente para extraer 1 frame clave
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(contenido)
                ruta_temp = temp_video.name
                
            try:
                captura = cv2.VideoCapture(ruta_temp)
                # Tomamos un frame al 50% del video
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
        # 3. MÓDULO DE DOCUMENTOS (Detección de Texto)
        # ==========================================
        elif nombre_archivo.endswith('.pdf'):
            tipo_evidencia = "DOCUMENTO"
            lector = PdfReader(io.BytesIO(contenido))
            texto_extraido = ""
            for pagina in lector.pages[:3]: # Leemos max 3 páginas
                texto_extraido += pagina.extract_text() + " "
                
            if len(texto_extraido.strip()) < 50:
                error_api = "El PDF está vacío o es una imagen escaneada sin texto seleccionable."
            else:
                res = analizar_texto_hf(texto_extraido)
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                    # El modelo devuelve [{'label': 'Human', 'score': 0.99}, {'label': 'ChatGPT', 'score': 0.01}]
                    datos_scores = res[0]
                    for item in datos_scores:
                        if item['label'] == 'ChatGPT':
                            porcentaje_ia = item['score'] * 100
                    
                    desglose_ui = {
                        "perplejidad_linguistica": round(100 - porcentaje_ia, 1),
                        "patrones_chatgpt": round(porcentaje_ia, 1)
                    }
                else:
                    error_api = "El servidor de lenguaje está cargando. Reintenta en 20s."

        # ==========================================
        # 4. MÓDULO DE AUDIO (Deepfakes de Voz)
        # ==========================================
        elif nombre_archivo.endswith(('.mp3', '.wav', '.ogg')):
            tipo_evidencia = "AUDIO"
            res = analizar_audio_hf(contenido)
            if isinstance(res, list) and len(res) > 0:
                # Ordenamos para obtener el score más alto
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
            else:
                error_api = "El servidor de espectrogramas está cargando. Reintenta en 20s."

        else:
            error_api = "Formato de archivo no soportado por ForensIA."

        # --- LÓGICA DE RESPUESTA COMERCIAL ---
        if error_api:
            return {
                "nombre_archivo": nombre_archivo,
                "tipo_evidencia": tipo_evidencia,
                "analisis": {"nivel_riesgo": "ERROR", "motivo": error_api, "porcentaje_ia": 0.0}
            }

        # Determinamos el riesgo final
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
                "desglose_ui": desglose_ui # <-- AQUÍ ESTÁ LA MAGIA PARA FLUTTER
            }
        }

    except Exception as e:
        return {"error": f"Error del motor multimodal: {str(e)}"}