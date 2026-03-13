from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS
from pypdf import PdfReader
import io

app = FastAPI() 

# --- DICCIONARIO DE FIRMAS DE IA ---
FIRMAS_IA = [
    "midjourney", 
    "dall-e", 
    "dall·e", 
    "stable diffusion", 
    "adobe firefly", 
    "generative fill",
    "ai generated",
    "comfyui"
]

def evaluar_presencia_ia(metadatos: dict) -> dict:
    """Busca firmas de motores de IA conocidos dentro de los metadatos."""
    # Convertimos todos los metadatos a texto minúscula para buscar fácilmente
    texto_evidencia = str(metadatos).lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_evidencia:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Se encontró la firma digital de '{firma.upper()}' en la evidencia."
            }
            
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "No se detectaron huellas conocidas de IA en los metadatos."
    }

@app.get("/")
def home():
    return {"mensaje": "API Forense con Metadatos e IA activa"}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    hash_resultado = hashlib.sha256(contenido).hexdigest()
    metadatos_extraidos = {}
    
    nombre_archivo = file.filename.lower() if file.filename else ""
    tipo_contenido = file.content_type if file.content_type else ""
    
    # 1. EXTRACCIÓN IMAGEN
    if tipo_contenido.startswith("image/") or nombre_archivo.endswith(('.png', '.jpg', '.jpeg')):
        try:
            imagen = Image.open(io.BytesIO(contenido))
            info_exif = imagen._getexif()
            if info_exif:
                for tag_id, valor in info_exif.items():
                    nombre_tag = TAGS.get(tag_id, tag_id)
                    metadatos_extraidos[nombre_tag] = str(valor)
            else:
                metadatos_extraidos = {"aviso": "Imagen sin metadatos EXIF"}
        except Exception as e:
            metadatos_extraidos = {"error": f"Error al extraer datos: {str(e)}"}

    # 2. EXTRACCIÓN PDF
    elif tipo_contenido == "application/pdf" or nombre_archivo.endswith('.pdf'):
        try:
            reader = PdfReader(io.BytesIO(contenido))
            info = reader.metadata
            if info:
                for key, value in info.items():
                    metadatos_extraidos[key.replace("/", "")] = str(value)
            else:
                metadatos_extraidos = {"aviso": "PDF sin metadatos internos"}
        except Exception as e:
            metadatos_extraidos = {"error": f"Error al extraer datos: {str(e)}"}
            
    else:
        metadatos_extraidos = {"aviso": "Tipo de archivo no soportado."}

    # --- NUEVO: EJECUTAR ANÁLISIS DE IA ---
    analisis_ia = evaluar_presencia_ia(metadatos_extraidos)

    return {
        "archivo": file.filename,
        "hash_sha256": hash_resultado,
        "evaluacion_ia": analisis_ia, # Añadimos el veredicto aquí
        "metadatos_completos": metadatos_extraidos
    }