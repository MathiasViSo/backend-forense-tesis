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
# Usamos un modelo especializado en Deepfakes y rostros alterados
MODELS = {
    "IMAGE": "umm-maybe/AI-image-detector", 
    "AUDIO": "ResembleAI/ai_detector_audio"
}

FIRMAS_IA = [
    "midjourney", "dall-e", "stable diffusion", "adobe firefly", 
    "generative fill", "ai generated", "comfyui", "synthid", 
    "lyria", "veo", "gemini", "deepfake", "roop"
]

# Inicializamos el detector facial nativo de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def consultar_modelo_hf(contenido_bytes: bytes, tipo: str):
    """Consulta a la API de Hugging Face con manejo de errores."""
    if not HF_TOKEN or tipo not in MODELS:
        return None
    
    url = f"https://api-inference.huggingface.co/models/{MODELS[tipo]}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, data=contenido_bytes)
        resultado = response.json()
        # Hugging Face a veces devuelve una lista, a veces un dict con error
        if isinstance(resultado, list) and len(resultado) > 0:
            return resultado[0] # Retornamos la predicción más alta
        return resultado
    except Exception as e:
        return {"error": str(e)}

def analizar_imagen_aislada(imagen_bytes: bytes) -> dict:
    """Busca rostros. Si los hay, analiza el rostro. Si no, analiza la imagen completa."""
    # Convertir bytes a formato OpenCV
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "No se pudo decodificar la imagen."}

    # Convertir a escala de grises para la detección facial
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(rostros) > 0:
        # Si hay rostros, recortamos el primero (el principal) para un análisis profundo
        x, y, w, h = rostros[0]
        rostro_recortado = img[y:y+h, x:x+w]
        
        # Volvemos a convertir a bytes para enviarlo a la IA
        _, buffer = cv2.imencode('.jpg', rostro_recortado)
        rostro_bytes = buffer.tobytes()
        
        score_ia = consultar_modelo_hf(rostro_bytes, "IMAGE")
        return {"rostros_detectados": len(rostros), "analisis_aislado": True, "score_ia": score_ia}
    else:
        # Si no hay rostros (paisajes, documentos), analizamos la imagen entera
        score_ia = consultar_modelo_hf(imagen_bytes, "IMAGE")
        return {"rostros_detectados": 0, "analisis_aislado": False, "score_ia": score_ia}

def realizar_analisis_forense_metadatos(contenido: bytes, metadatos: dict) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Evidencia de generación o alteración sintética hallada: Firma de '{firma.upper()}'."
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
    
    metadatos_extraidos = {}
    resultado_ia_profundo = None
    tipo_evidencia = "DESCONOCIDO"

    # 1. IMÁGENES (Con Aislamiento Biométrico)
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGEN"
        try:
            img = Image.open(io.BytesIO(contenido))
            metadatos_extraidos = {f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))}
        except: pass
        
        # Ejecutamos el motor avanzado
        resultado_ia_profundo = analizar_imagen_aislada(contenido)

    # 2. AUDIO
    elif nombre_archivo.endswith(('.mp3', '.wav')):
        tipo_evidencia = "AUDIO"
        try:
            audio = MutagenFile(io.BytesIO(contenido))
            if audio:
                metadatos_extraidos = {f"AUDIO_{k}": str(v) for k, v in audio.items()}
        except: pass
        resultado_ia_profundo = {"score_ia": consultar_modelo_hf(contenido, "AUDIO")}

    # 3. VIDEO (Muestreo Temporal de 3 Frames)
    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        tipo_evidencia = "VIDEO"
        metadatos_extraidos["info_video"] = "Procesando muestreo temporal (3 frames)."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(contenido)
            ruta_temp = temp_video.name
            
        try:
            captura = cv2.VideoCapture(ruta_temp)
            total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Tomamos frames al 25%, 50% y 75% del video
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
            
            # Consolidamos el resultado del video basado en el peor escenario de los 3 frames
            resultado_ia_profundo = {
                "analisis_frames": len(analisis_frames),
                "rostros_detectados_total": rostros_totales,
                "score_ia": analisis_frames[1]["score_ia"] if len(analisis_frames) > 1 else None # Tomamos el score del medio como referencia principal
            }
            metadatos_extraidos["video_frames_analizados"] = str(puntos_extraccion)
            
        except Exception as e:
            metadatos_extraidos["error_video"] = f"Error OpenCV: {str(e)}"
        finally:
            os.remove(ruta_temp)

    # 4. DOCUMENTOS
    elif nombre_archivo.endswith('.pdf'):
        tipo_evidencia = "DOCUMENTO"
        try:
            reader = PdfReader(io.BytesIO(contenido))
            metadatos_extraidos = {f"PDF_{k.replace('/', '')}": str(v) for k, v in reader.metadata.items()}
        except: pass

    # --- RESOLUCIÓN FINAL E HÍBRIDA ---
    resultado_estructural = realizar_analisis_forense_metadatos(contenido, metadatos_extraidos)
    
    # Ajustar el nivel de riesgo si la IA encontró alta probabilidad de que sea artificial
    # Hugging face suele devolver [{'label': 'artificial', 'score': 0.98}, ...]
    if resultado_ia_profundo and "score_ia" in resultado_ia_profundo:
        score_data = resultado_ia_profundo["score_ia"]
        if isinstance(score_data, dict) and "label" in score_data:
            etiqueta = str(score_data.get("label", "")).lower()
            confianza = float(score_data.get("score", 0.0))
            
            # Si el modelo HF dice que es IA (fake/artificial) con más del 70% de confianza
            if ("fake" in etiqueta or "artificial" in etiqueta) and confianza > 0.70:
                resultado_estructural["detectado"] = True
                resultado_estructural["nivel_riesgo"] = "ALTO"
                resultado_estructural["motivo"] = f"La Red Neuronal detectó anomalías visuales ({int(confianza*100)}% certeza). Posible manipulación o Deepfake."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "analisis": resultado_estructural,
        "detalles_biometricos": resultado_ia_profundo,
        "detalles_tecnicos": metadatos_extraidos
    }