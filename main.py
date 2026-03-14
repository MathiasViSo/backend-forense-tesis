from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from pypdf import PdfReader
from mutagen import File as MutagenFile
import io
import os
import requests
import cv2
import numpy as np
import tempfile

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

def consultar_modelo_hf(contenido_bytes: bytes, tipo: str):
    if not HF_TOKEN or tipo not in MODELS:
        return None
    url = f"https://api-inference.huggingface.co/models/{MODELS[tipo]}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(url, headers=headers, data=contenido_bytes)
        resultado = response.json()
        if isinstance(resultado, list) and len(resultado) > 0:
            return resultado[0] 
        return resultado
    except Exception as e:
        return {"error": str(e)}

def analizar_imagen_aislada(imagen_bytes: bytes) -> dict:
    """Clasifica entre Análisis Biométrico (Rostros) y Análisis de Objetos/Superficies."""
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "No se pudo decodificar la imagen."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(rostros) > 0:
        # MODO 1: Hay personas. Hacemos aislamiento biométrico.
        x, y, w, h = rostros[0]
        rostro_recortado = img[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.jpg', rostro_recortado)
        rostro_bytes = buffer.tobytes()
        
        score_ia = consultar_modelo_hf(rostro_bytes, "IMAGE")
        return {
            "modo_analisis": "BIOMÉTRICO",
            "rostros_detectados": len(rostros), 
            "score_ia": score_ia
        }
    else:
        # MODO 2: No hay rostros. Podría ser un producto de Marketplace, paisaje o textura.
        score_ia = consultar_modelo_hf(imagen_bytes, "IMAGE")
        return {
            "modo_analisis": "OBJETOS_Y_SUPERFICIES",
            "rostros_detectados": 0, 
            "score_ia": score_ia
        }

def realizar_analisis_forense_metadatos(contenido: bytes, metadatos: dict, es_captura: bool) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Evidencia de generación o alteración sintética hallada: Firma de '{firma.upper()}'."
            }
            
    if es_captura:
        return {
            "detectado": False,
            "nivel_riesgo": "AMBIGUO",
            "motivo": "Los metadatos han sido purgados (Posible Captura de Pantalla/Red Social). El veredicto dependerá exclusivamente del análisis visual."
        }
    
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "Estructura binaria limpia y sin firmas conocidas."
    }

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    # 1. DETECCIÓN DE CAPTURAS DE PANTALLA Y REDES SOCIALES
    palabras_captura = ['screenshot', 'captura', 'whatsapp', 'screen_', 'image-', 'img-']
    es_captura = any(p in nombre_archivo for p in palabras_captura)
    
    metadatos_extraidos = {}
    resultado_ia_profundo = None
    tipo_evidencia = "DESCONOCIDO"

    if es_captura:
        metadatos_extraidos["ALERTA_FORENSE"] = "El archivo parece ser una captura de pantalla o descarga de red social. Cadena de custodia de metadatos rota."

    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGEN"
        try:
            img = Image.open(io.BytesIO(contenido))
            metadatos_extraidos.update({f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))})
        except: pass
        
        resultado_ia_profundo = analizar_imagen_aislada(contenido)

    elif nombre_archivo.endswith(('.mp3', '.wav')):
        tipo_evidencia = "AUDIO"
        try:
            audio = MutagenFile(io.BytesIO(contenido))
            if audio:
                metadatos_extraidos.update({f"AUDIO_{k}": str(v) for k, v in audio.items()})
        except: pass
        resultado_ia_profundo = {"modo_analisis": "ESPECTROGRAMA", "score_ia": consultar_modelo_hf(contenido, "AUDIO")}

    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        tipo_evidencia = "VIDEO"
        metadatos_extraidos["info_video"] = "Procesando muestreo temporal (3 frames)."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(contenido)
            ruta_temp = temp_video.name
            
        try:
            captura = cv2.VideoCapture(ruta_temp)
            total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            puntos_extraccion = [int(total_frames * 0.25), int(total_frames * 0.50), int(total_frames * 0.75)]
            analisis_frames = []
            rostros_totales = 0

            for frame_idx in puntos_extraccion:
                captura.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                exito, frame = captura.read()
                if exito:
                    _, buffer = cv2.imencode('.jpg', frame)
                    resultado_frame = analizar_imagen_aislada(buffer.tobytes())
                    analisis_frames.append(resultado_frame)
                    rostros_totales += resultado_frame.get("rostros_detectados", 0)

            captura.release()
            
            # Si al menos un frame tuvo rostros, lo clasificamos como biométrico, sino como objetos.
            modo_predominante = "BIOMÉTRICO" if rostros_totales > 0 else "OBJETOS_Y_SUPERFICIES"
            
            resultado_ia_profundo = {
                "modo_analisis": modo_predominante,
                "analisis_frames": len(analisis_frames),
                "rostros_detectados_total": rostros_totales,
                "score_ia": analisis_frames[1]["score_ia"] if len(analisis_frames) > 1 else None
            }
        except Exception as e:
            metadatos_extraidos["error_video"] = f"Error OpenCV: {str(e)}"
        finally:
            os.remove(ruta_temp)

    elif nombre_archivo.endswith('.pdf'):
        tipo_evidencia = "DOCUMENTO"
        try:
            reader = PdfReader(io.BytesIO(contenido))
            metadatos_extraidos.update({f"PDF_{k.replace('/', '')}": str(v) for k, v in reader.metadata.items()})
        except: pass

    # --- RESOLUCIÓN FINAL E HÍBRIDA ---
    resultado_estructural = realizar_analisis_forense_metadatos(contenido, metadatos_extraidos, es_captura)
    
    # Evaluación contundente de la IA visual (incluso si los metadatos estaban limpios)
    if resultado_ia_profundo and "score_ia" in resultado_ia_profundo and resultado_ia_profundo["score_ia"]:
        score_data = resultado_ia_profundo["score_ia"]
        if isinstance(score_data, dict) and "label" in score_data:
            etiqueta = str(score_data.get("label", "")).lower()
            confianza = float(score_data.get("score", 0.0))
            
            if ("fake" in etiqueta or "artificial" in etiqueta) and confianza > 0.65:
                resultado_estructural["detectado"] = True
                resultado_estructural["nivel_riesgo"] = "ALTO"
                
                # Explicación adaptada al caso de uso (Rostro vs Objeto)
                modo = resultado_ia_profundo.get("modo_analisis", "")
                if modo == "OBJETOS_Y_SUPERFICIES":
                    resultado_estructural["motivo"] = f"Riesgo de Inpainting o generación sintética de entorno/producto ({int(confianza*100)}% certeza visual)."
                else:
                    resultado_estructural["motivo"] = f"La Red Neuronal detectó anomalías visuales ({int(confianza*100)}% certeza). Posible manipulación."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "es_captura": es_captura,
        "analisis": resultado_estructural,
        "detalles_biometricos": resultado_ia_profundo,
        "detalles_tecnicos": metadatos_extraidos
    }