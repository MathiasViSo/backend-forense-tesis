from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from pypdf import PdfReader
from docx import Document
import io

app = FastAPI()

# Diccionario expandido de firmas de IA y software de edición
FIRMAS_IA = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", 
    "adobe firefly", "generative fill", "ai generated", 
    "comfyui", "bing", "microsoft", "metadata", "canva",
    "photoshop", "gimp", "diffusion", "krea"
]

def realizar_analisis_forense(contenido: bytes, metadatos: dict) -> dict:
    # 1. Análisis sobre metadatos estructurados
    texto_metadatos = str(metadatos).lower()
    
    # 2. Análisis de "Strings" (Búsqueda binaria profunda)
    # Esto es clave para detectar firmas ocultas en archivos "limpios"
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Firma detectada: '{firma.upper()}' hallada en la estructura del archivo."
            }
    
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "No se detectaron huellas digitales de IA conocidas."
    }

@app.get("/")
def home():
    return {"status": "ForensIA API Activa", "analisis_disponibles": ["Imagen", "PDF", "Word"]}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    metadatos_extraidos = {}

    # --- PROCESAMIENTO SEGÚN TIPO DE ARCHIVO ---
    
    # 1. IMÁGENES (JPG, PNG, WEBP, etc.)
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff')):
        try:
            img = Image.open(io.BytesIO(contenido))
            if img.info:
                for k, v in img.info.items():
                    if isinstance(v, (str, bytes)):
                        metadatos_extraidos[f"IMG_{k}"] = str(v)
            # Intentar obtener EXIF si existe
            exif = img.getexif()
            if exif:
                for tag, value in exif.items():
                    metadatos_extraidos[f"EXIF_{tag}"] = str(value)
        except Exception as e:
            metadatos_extraidos["error_img"] = f"Fallo al leer imagen: {str(e)}"

    # 2. PDF
    elif nombre_archivo.endswith('.pdf'):
        try:
            reader = PdfReader(io.BytesIO(contenido))
            info = reader.metadata
            if info:
                for key, value in info.items():
                    metadatos_extraidos[f"PDF_{key.replace('/', '')}"] = str(value)
        except Exception as e:
            metadatos_extraidos["error_pdf"] = f"Fallo al leer PDF: {str(e)}"

    # 3. WORD (DOCX)
    elif nombre_archivo.endswith('.docx'):
        try:
            doc = Document(io.BytesIO(contenido))
            props = doc.core_properties
            metadatos_extraidos = {
                "autor": props.author,
                "creado": str(props.created),
                "modificado": str(props.modified),
                "ultima_modificacion_por": props.last_modified_by,
                "software_creador": "Microsoft Word / Office"
            }
        except Exception as e:
            metadatos_extraidos["error_docx"] = f"Fallo al leer Word: {str(e)}"

    # Ejecutar el análisis forense basado en lo recolectado
    resultado_ia = realizar_analisis_forense(contenido, metadatos_extraidos)
    
    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "analisis": resultado_ia,
        "detalles_tecnicos": metadatos_extraidos if metadatos_extraidos else {"aviso": "Sin metadatos legibles"}
    }