from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
import io

app = FastAPI()

# Lista expandida de firmas de IA (incluyendo metadatos XMP y bloques tEXt)
FIRMAS_IA = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", 
    "adobe firefly", "generative fill", "ai generated", 
    "comfyui", "bing", "microsoft", "metadata"
]

def realizar_analisis_forense(contenido: bytes, metadatos: dict) -> dict:
    # 1. Convertir todo a minúsculas para búsqueda insensible a mayúsculas
    texto_metadatos = str(metadatos).lower()
    # 2. Análisis de "Strings" (Cadenas crudas en el binario)
    # Esto detecta firmas que PIL a veces ignora
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Evidencia hallada: Firma de '{firma.upper()}' en la estructura interna."
            }
    
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "No se encontraron patrones de IA conocidos."
    }

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    metadatos_extraidos = {}
    
    # Intento de extracción con PIL (para imágenes)
    try:
        img = Image.open(io.BytesIO(contenido))
        # Extraer TODO lo que PIL pueda leer (incluyendo info y exif)
        if img.info:
            for k, v in img.info.items():
                if isinstance(v, (str, bytes)):
                    metadatos_extraidos[f"INFO_{k}"] = str(v)
    except:
        pass

    # Ejecutar el análisis forense
    resultado_ia = realizar_analisis_forense(contenido, metadatos_extraidos)
    
    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "analisis": resultado_ia,
        "detalles_tecnicos": metadatos_extraidos
    }