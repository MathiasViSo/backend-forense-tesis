from fastapi import FastAPI, File, UploadFile
from huggingface_hub import InferenceClient
import hashlib
from PIL import Image, ImageChops, ImageEnhance
from pypdf import PdfReader
from mutagen import File as MutagenFile
import io
import os
import cv2
import numpy as np
import base64
import gc

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

# --- EL NUEVO MOTOR PROFESIONAL DE HUGGING FACE ---
# Inicializamos el cliente oficial que maneja colas, timeouts y retries automáticamente
try:
    hf_client = InferenceClient(token=HF_TOKEN)
except Exception as e:
    print(f"Error iniciando cliente HF: {e}")
    hf_client = None

# Los dos mejores modelos de detección de la capa gratuita
MODELO_IMAGEN = "dima806/ai_vs_real_image_detection"
MODELO_AUDIO = "ResembleAI/ai_detector_audio"

FIRMAS_IA = [
    "midjourney", "dall-e", "stable diffusion", "adobe firefly", 
    "generative fill", "ai generated", "comfyui", "synthid", 
    "lyria", "veo", "gemini", "deepfake", "roop", "bing"
]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def consultar_ia_profesional(contenido_bytes: bytes, tipo: str):
    """Consulta usando el SDK oficial de Hugging Face Hub"""
    if not hf_client:
        return {"error": "El cliente de Hugging Face no pudo inicializarse. Revisa el Token."}
        
    try:
        if tipo == "IMAGE":
            # El SDK requiere un objeto Image de PIL, no bytes crudos
            imagen = Image.open(io.BytesIO(contenido_bytes)).convert("RGB")
            
            # .image_classification maneja el "despertar" del modelo de forma nativa
            resultados = hf_client.image_classification(imagen, model=MODELO_IMAGEN)
            
            # Convertimos la respuesta del SDK a nuestra lista de diccionarios
            return [{"label": res.label, "score": res.score} for res in resultados]
            
        elif tipo == "AUDIO":
            # Para audio, el cliente permite mandar los bytes directos
            resultados = hf_client.audio_classification(contenido_bytes, model=MODELO_AUDIO)
            return [{"label": res.label, "score": res.score} for res in resultados]
            
    except Exception as e:
        error_msg = str(e).lower()
        if "503" in error_msg or "loading" in error_msg:
            return {"error": "La Red Neuronal se está encendiendo. Por favor, intenta de nuevo en 30 segundos."}
        return {"error": f"Fallo en SDK de Hugging Face: {str(e)}"}

# --- MOTOR FORENSE ELA ---
def aplicar_analisis_ela(contenido_imagen: bytes, calidad_recompresion: int = 90) -> dict:
    try:
        imagen_original = Image.open(io.BytesIO(contenido_imagen)).convert('RGB')
        buffer_temporal = io.BytesIO()
        imagen_original.save(buffer_temporal, 'JPEG', quality=calidad_recompresion)
        buffer_temporal.seek(0)
        imagen_recomprimida = Image.open(buffer_temporal)
        
        diferencia_ela = ImageChops.difference(imagen_original, imagen_recomprimida)
        ela_array = np.array(diferencia_ela)
        
        desviacion_ruido = float(np.std(ela_array))
        sospecha_manipulacion = desviacion_ruido > 12.0 
        
        extremos = diferencia_ela.getextrema()
        max_diff = max([ex[1] for ex in extremos])
        if max_diff == 0: max_diff = 1
        escala_brillo = 255.0 / max_diff
        imagen_ela_resaltada = ImageEnhance.Brightness(diferencia_ela).enhance(escala_brillo)
        
        buffer_salida = io.BytesIO()
        imagen_ela_resaltada.save(buffer_salida, format="JPEG")
        mapa_calor_b64 = base64.b64encode(buffer_salida.getvalue()).decode("utf-8")
        
        return {
            "ela_ejecutado": True,
            "desviacion_estandar_ruido": round(desviacion_ruido, 2),
            "anomalia_detectada_ela": sospecha_manipulacion,
            "mapa_calor_base64": mapa_calor_b64
        }
    except Exception as e:
        return {"ela_ejecutado": False, "error_ela": str(e)}

def analizar_imagen_aislada(imagen_bytes: bytes) -> dict:
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Error de decodificación."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(rostros) > 0:
        x, y, w, h = rostros[0]
        rostro_recortado = img[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.jpg', rostro_recortado)
        score_ia = consultar_ia_profesional(buffer.tobytes(), "IMAGE")
        return {"modo_analisis": "BIOMÉTRICO", "rostros_detectados": len(rostros), "score_ia": score_ia}
    else:
        score_ia = consultar_ia_profesional(imagen_bytes, "IMAGE")
        return {"modo_analisis": "OBJETOS_Y_SUPERFICIES", "rostros_detectados": 0, "score_ia": score_ia}

def realizar_analisis_forense_metadatos(contenido: bytes, metadatos: dict, es_captura: bool) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {"detectado": True, "nivel_riesgo": "ALTO", "motivo": f"Firma de '{firma.upper()}' detectada en código base."}
    if es_captura:
        return {"detectado": False, "nivel_riesgo": "PREVENTIVO", "motivo": "Metadatos purgados (Captura/WhatsApp)."}
    return {"detectado": False, "nivel_riesgo": "BAJO", "motivo": "Sin firmas conocidas."}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    es_captura = any(p in nombre_archivo for p in ['screenshot', 'captura', 'whatsapp', 'screen_', 'image-', 'img-'])
    
    metadatos_extraidos = {}
    resultado_ia_profundo = None
    resultado_ela = None
    tipo_evidencia = "DESCONOCIDO"

    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGEN"
        try:
            resultado_ela = aplicar_analisis_ela(contenido)
            if resultado_ela and resultado_ela.get("ela_ejecutado"):
                metadatos_extraidos["analisis_ela_matematico"] = resultado_ela

            img = Image.open(io.BytesIO(contenido)).convert('RGB')
            metadatos_extraidos.update({f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))})
            
            img.thumbnail((800, 800)) 
            buffer_optimo = io.BytesIO()
            img.save(buffer_optimo, format="JPEG", quality=95)
            contenido_optimo = buffer_optimo.getvalue()
            
            resultado_ia_profundo = analizar_imagen_aislada(contenido_optimo)
            del img, buffer_optimo
            gc.collect() 
        except Exception as e: print(f"Error procesando imagen: {e}")

    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        tipo_evidencia = "VIDEO"
        resultado_ia_profundo = {"modo_analisis": "VIDEO_FRAMES", "score_ia": []}
        
    elif nombre_archivo.endswith(('.mp3', '.wav')):
        tipo_evidencia = "AUDIO"
        try:
            audio = MutagenFile(io.BytesIO(contenido))
            if audio:
                metadatos_extraidos.update({f"AUDIO_{k}": str(v) for k, v in audio.items()})
        except: pass
        resultado_ia_profundo = {"modo_analisis": "ESPECTROGRAMA", "score_ia": consultar_ia_profesional(contenido, "AUDIO")}

    # --- LÓGICA DE PORCENTAJE ---
    resultado_estructural = realizar_analisis_forense_metadatos(contenido, metadatos_extraidos, es_captura)
    porcentaje_ia = 0.0
    error_ia = None

    if tipo_evidencia in ["IMAGEN", "AUDIO"] and resultado_ia_profundo:
        score_data = resultado_ia_profundo.get("score_ia")
        
        if isinstance(score_data, list) and len(score_data) > 0:
            score_data = sorted(score_data, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            top_label = str(score_data[0].get("label", "")).lower()
            top_score = float(score_data[0].get("score", 0.0))
            
            if any(x in top_label for x in ["human", "hum", "real", "genuine", "0"]):
                porcentaje_ia = (1.0 - top_score) * 100
            else:
                porcentaje_ia = top_score * 100
                
        elif isinstance(score_data, dict) and "error" in score_data:
            error_ia = score_data["error"]

        anomalia_ela = resultado_ela.get("anomalia_detectada_ela", False) if resultado_ela else False
        resultado_estructural["porcentaje_ia"] = round(porcentaje_ia, 2)

        if error_ia:
            resultado_estructural["nivel_riesgo"] = "AMBIGUO"
            resultado_estructural["motivo"] = error_ia
        else:
            if porcentaje_ia > 60.0:
                resultado_estructural["nivel_riesgo"] = "ALTO"
                resultado_estructural["motivo"] = "Análisis de Red Neuronal detecta alta probabilidad de origen sintético."
            elif porcentaje_ia > 20.0 or anomalia_ela:
                resultado_estructural["nivel_riesgo"] = "PREVENTIVO"
                resultado_estructural["motivo"] = "Se detectan trazas de IA o alteraciones matemáticas (ELA)."
            else:
                resultado_estructural["nivel_riesgo"] = "BAJO"
                resultado_estructural["motivo"] = "Estructura correspondiente a origen humano/cámara."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "es_captura": es_captura,
        "analisis": resultado_estructural,
        "detalles_biometricos": resultado_ia_profundo,
        "detalles_tecnicos": metadatos_extraidos
    }