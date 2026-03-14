from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from pypdf import PdfReader
from docx import Document
from mutagen import File as MutagenFile
import io
import os
import requests
import cv2
import tempfile

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS = {
    "IMAGE": "umm-maybe/AI-image-detector",
    "AUDIO": "ResembleAI/ai_detector_audio"
}

# Diccionario de firmas expandido (¡Ahora incluye SynthID de Google/Gemini!)
FIRMAS_IA = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", 
    "adobe firefly", "generative fill", "ai generated", 
    "comfyui", "bing", "microsoft", "metadata", "canva",
    "photoshop", "gimp", "diffusion", "krea", "elevenlabs", 
    "vall-e", "synthid", "lyria", "veo", "gemini", "google inc"
]

def consultar_modelo_hf(contenido_archivo: bytes, tipo: str):
    if not HF_TOKEN or tipo not in MODELS:
        return None
    
    url = f"https://api-inference.huggingface.co/models/{MODELS[tipo]}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, data=contenido_archivo)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def realizar_analisis_forense(contenido: bytes, metadatos: dict) -> dict:
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
        "motivo": "Estructura binaria limpia. No se hallaron firmas de IA generativa conocidas."
    }

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    metadatos_extraidos = {}
    tipo_evidencia = None
    score_ia = None

    # 1. IMÁGENES
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGE"
        try:
            img = Image.open(io.BytesIO(contenido))
            metadatos_extraidos = {f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))}
        except: pass

    # 2. AUDIO (MP3, WAV)
    elif nombre_archivo.endswith(('.mp3', '.wav')):
        tipo_evidencia = "AUDIO"
        try:
            audio = MutagenFile(io.BytesIO(contenido))
            if audio:
                metadatos_extraidos = {f"AUDIO_{k}": str(v) for k, v in audio.items()}
        except: pass

    # 3. VIDEO (Análisis por extracción de Frames)
    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        metadatos_extraidos["info_video"] = "Procesando frames mediante OpenCV."
        
        # OpenCV necesita un archivo físico, así que guardamos el video temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(contenido)
            ruta_temp = temp_video.name
            
        try:
            captura = cv2.VideoCapture(ruta_temp)
            # Extraemos un frame de la mitad del video donde suele estar la acción principal
            total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            captura.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            exito, frame = captura.read()
            
            if exito:
                # Convertimos el frame (imagen) a bytes para enviarlo a Hugging Face
                exito_codificacion, buffer = cv2.imencode('.jpg', frame)
                if exito_codificacion:
                    frame_bytes = buffer.tobytes()
                    # Analizamos el frame extraído como si fuera una imagen
                    score_ia = consultar_modelo_hf(frame_bytes, "IMAGE")
                    metadatos_extraidos["analisis_video"] = "Frame intermedio analizado con éxito por modelo de visión."
            captura.release()
        except Exception as e:
            metadatos_extraidos["error_video"] = f"Error al extraer frames: {str(e)}"
        finally:
            os.remove(ruta_temp) # Limpiamos el servidor

    # 4. DOCUMENTOS (PDF/Word) - Omitidos por brevedad, usa la lógica que ya teníamos.
    elif nombre_archivo.endswith('.pdf'):
        try:
            reader = PdfReader(io.BytesIO(contenido))
            metadatos_extraidos = {f"PDF_{k.replace('/', '')}": str(v) for k, v in reader.metadata.items()}
        except: pass

    # --- RESOLUCIÓN FINAL ---
    resultado_forense = realizar_analisis_forense(contenido, metadatos_extraidos)
    
    # Si es imagen o audio, consultamos HF aquí (video ya se consultó en su bloque)
    if tipo_evidencia in ["IMAGE", "AUDIO"] and score_ia is None:
        score_ia = consultar_modelo_hf(contenido, tipo_evidencia)

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "analisis": resultado_forense,
        "score_ia_huggingface": score_ia,
        "detalles_tecnicos": metadatos_extraidos
    }