from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image, ImageChops, ImageEnhance
from pypdf import PdfReader
from mutagen import File as MutagenFile
import io
import os
import requests
import cv2
import numpy as np
import tempfile
import base64
import time
import gc

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODELS = {
    "IMAGE": "umm-maybe/AI-image-detector", 
    "AUDIO": "ResembleAI/ai_detector_audio"
}

FIRMAS_IA = [
    "midjourney", "dall-e", "stable diffusion", "adobe firefly", 
    "generative fill", "ai generated", "comfyui", "synthid", 
    "lyria", "veo", "gemini", "deepfake", "roop", "bing"
]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def consultar_modelo_hf(contenido_bytes: bytes, tipo: str, max_intentos=3):
    if not HF_TOKEN or tipo not in MODELS:
        return {"error": "Token de Hugging Face no configurado."}
    
    url = f"https://api-inference.huggingface.co/models/{MODELS[tipo]}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for intento in range(max_intentos):
        try:
            response = requests.post(url, headers=headers, data=contenido_bytes)
            resultado = response.json()
            
            if isinstance(resultado, dict) and "estimated_time" in resultado:
                tiempo_espera = resultado["estimated_time"]
                print(f"[{tipo}] Modelo durmiendo. Esperando {tiempo_espera}s...")
                time.sleep(tiempo_espera + 2)
                continue
            
            if isinstance(resultado, dict) and "error" in resultado:
                return {"error": resultado["error"]}
                
            # Corrección vital: Devolver la lista completa para buscar el porcentaje exacto
            if isinstance(resultado, list):
                # HuggingFace a veces anida las listas [[{...}, {...}]]
                if len(resultado) > 0 and isinstance(resultado[0], list):
                    return resultado[0]
                return resultado
                
            return resultado
        except Exception as e:
            return {"error": f"Fallo de conexión: {str(e)}"}
            
    return {"error": "El servidor de IA no respondió a tiempo."}

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
    
    if img is None:
        return {"error": "No se pudo decodificar la imagen."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(rostros) > 0:
        x, y, w, h = rostros[0]
        rostro_recortado = img[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.jpg', rostro_recortado)
        score_ia = consultar_modelo_hf(buffer.tobytes(), "IMAGE")
        return {"modo_analisis": "BIOMÉTRICO", "rostros_detectados": len(rostros), "score_ia": score_ia}
    else:
        score_ia = consultar_modelo_hf(imagen_bytes, "IMAGE")
        return {"modo_analisis": "OBJETOS_Y_SUPERFICIES", "rostros_detectados": 0, "score_ia": score_ia}

def realizar_analisis_forense_metadatos(contenido: bytes, metadatos: dict, es_captura: bool) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {"detectado": True, "nivel_riesgo": "ALTO", "motivo": f"Firma oculta de '{firma.upper()}' detectada."}
            
    if es_captura:
        return {"detectado": False, "nivel_riesgo": "PREVENTIVO", "motivo": "Metadatos purgados (Posible Captura/WhatsApp)."}
    
    return {"detectado": False, "nivel_riesgo": "BAJO", "motivo": "Sin firmas conocidas."}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    palabras_captura = ['screenshot', 'captura', 'whatsapp', 'screen_', 'image-', 'img-']
    es_captura = any(p in nombre_archivo for p in palabras_captura)
    
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

        except Exception as e: 
            print(f"Error procesando imagen: {e}")

    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        tipo_evidencia = "VIDEO"
        metadatos_extraidos["info_video"] = "Muestreo temporal de frames."
        # ... (Mantén tu lógica de video aquí si la necesitas, simplificada para el ejemplo)
        resultado_ia_profundo = {"modo_analisis": "VIDEO_FRAMES", "score_ia": []}

    # --- EXTRACCIÓN DEL PORCENTAJE EXACTO DE IA ---
    resultado_estructural = realizar_analisis_forense_metadatos(contenido, metadatos_extraidos, es_captura)
    porcentaje_ia = 0.0
    error_ia = None

    if tipo_evidencia == "IMAGEN" and resultado_ia_profundo:
        score_data = resultado_ia_profundo.get("score_ia")
        
        # Buscamos en TODA la lista la etiqueta de falsificación
        if isinstance(score_data, list):
            for item in score_data:
                lbl = str(item.get("label", "")).lower()
                val = float(item.get("score", 0.0))
                if any(x in lbl for x in ["fake", "artificial", "generated", "ai"]):
                    porcentaje_ia = val * 100
                    break
        elif isinstance(score_data, dict) and "error" in score_data:
            error_ia = score_data["error"]

        anomalia_ela = resultado_ela.get("anomalia_detectada_ela", False) if resultado_ela else False

        # Inyectamos el porcentaje en la respuesta
        resultado_estructural["porcentaje_ia"] = round(porcentaje_ia, 2)

        if error_ia:
            resultado_estructural["nivel_riesgo"] = "AMBIGUO"
            resultado_estructural["motivo"] = f"Error de IA: {error_ia}. ELA Anomalía: {anomalia_ela}"
        else:
            # Determinamos el nivel de riesgo en base al porcentaje
            if porcentaje_ia > 65.0:
                resultado_estructural["nivel_riesgo"] = "ALTO"
                resultado_estructural["motivo"] = "Análisis visual detecta alta probabilidad de manipulación o síntesis."
            elif porcentaje_ia > 15.0 or anomalia_ela:
                resultado_estructural["nivel_riesgo"] = "PREVENTIVO"
                resultado_estructural["motivo"] = "Se detectan trazas de IA o alteraciones de compresión (ELA). Requiere revisión."
            else:
                resultado_estructural["nivel_riesgo"] = "BAJO"
                resultado_estructural["motivo"] = "Alta probabilidad de origen humano/cámara."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "es_captura": es_captura,
        "analisis": resultado_estructural,
        "detalles_biometricos": resultado_ia_profundo,
        "detalles_tecnicos": metadatos_extraidos
    }