from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from pypdf import PdfReader
from docx import Document
from mutagen import File as MutagenFile
import io
import os
import requests

app = FastAPI()

# Configuración de Hugging Face (Asegúrate de poner tu Token en Render -> Environment Variables)
HF_TOKEN = os.getenv("HF_TOKEN")
# Modelo sugerido para detección de imágenes IA
MODEL_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

FIRMAS_IA = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", 
    "adobe firefly", "generative fill", "ai generated", 
    "comfyui", "bing", "microsoft", "metadata", "canva",
    "photoshop", "gimp", "diffusion", "krea", "elevenlabs"
]

def consultar_modelo_hf(contenido_archivo: bytes):
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(MODEL_URL, headers=headers, data=contenido_archivo)
        return response.json()
    except:
        return None

def realizar_analisis_forense(contenido: bytes, metadatos: dict) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Evidencia de manipulación o generación por IA: '{firma.upper()}'."
            }
    
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "No se encontraron firmas de IA conocidas en la estructura binaria."
    }

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    metadatos_extraidos = {}

    # 1. IMÁGENES
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        try:
            img = Image.open(io.BytesIO(contenido))
            metadatos_extraidos = {f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))}
        except: pass

    # 2. PDF
    elif nombre_archivo.endswith('.pdf'):
        try:
            reader = PdfReader(io.BytesIO(contenido))
            metadatos_extraidos = {f"PDF_{k.replace('/', '')}": str(v) for k, v in reader.metadata.items()}
        except: pass

    # 3. WORD
    elif nombre_archivo.endswith('.docx'):
        try:
            doc = Document(io.BytesIO(contenido))
            metadatos_extraidos = {"autor": doc.core_properties.author, "software": "MS Word"}
        except: pass

    # 4. AUDIO (MP3, WAV)
    elif nombre_archivo.endswith(('.mp3', '.wav')):
        try:
            audio = MutagenFile(io.BytesIO(contenido))
            if audio:
                metadatos_extraidos = {f"AUDIO_{k}": str(v) for k, v in audio.items()}
        except: pass

    # 5. VIDEO (Metadatos básicos por extensión)
    elif nombre_archivo.endswith(('.mp4', '.avi', '.mov')):
        metadatos_extraidos = {"info": "Archivo de video detectado", "analisis": "Requiere peritaje de frames"}

    resultado_forense = realizar_analisis_forense(contenido, metadatos_extraidos)
    
    # Inferencia opcional si es imagen
    score_ia = None
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg')):
        score_ia = consultar_modelo_hf(contenido)

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "analisis": resultado_forense,
        "score_ia_huggingface": score_ia,
        "detalles_tecnicos": metadatos_extraidos
    }