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
    "comfyui",
    "bing" # Aquí está Bing incluido
]

# --- 1. FUNCIÓN DE DETECCIÓN (Con escaneo binario en crudo) ---
def evaluar_presencia_ia(metadatos: dict, contenido_crudo: bytes) -> dict:
    """Busca firmas de motores de IA en los metadatos y en el binario crudo."""
    texto_evidencia = str(metadatos).lower()
    
    # Técnica forense: Cadenas en crudo (Strings)
    texto_binario = contenido_crudo.decode('utf-8', errors='ignore').lower()
    
    for firma in FIRMAS_IA:
        # Buscamos en la evidencia formal o en la estructura profunda
        if firma in texto_evidencia or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "motivo": f"Se encontró la firma digital de '{firma.upper()}' oculta en la estructura del archivo."
            }
            
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "motivo": "No se detectaron huellas conocidas de IA en la estructura."
    }

# --- 2. RUTAS DE LA API ---
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
    
    # ---------------------------------------------------------
    # ANÁLISIS DE IMÁGENES
    # ---------------------------------------------------------
    if tipo_contenido.startswith("image/") or nombre_archivo.endswith(('.png', '.jpg', '.jpeg')):
        try:
            imagen = Image.open(io.BytesIO(contenido))
            
            # A) Buscar metadatos EXIF
            if hasattr(imagen, '_getexif') and imagen._getexif():
                for tag_id, valor in imagen._getexif().items():
                    nombre_tag = TAGS.get(tag_id, tag_id)
                    metadatos_extraidos[f"EXIF_{nombre_tag}"] = str(valor)
            
            # B) Buscar metadatos PNG (Text Chunks)
            if hasattr(imagen, 'info') and imagen.info:
                for clave, valor in imagen.info.items():
                    if isinstance(valor, str) or isinstance(valor, bytes):
                        metadatos_extraidos[f"PNG_{clave}"] = str(valor)

            if not metadatos_extraidos:
                metadatos_extraidos = {"aviso": "Imagen completamente limpia (sin EXIF ni PNG Chunks)"}
                
        except Exception as e:
            metadatos_extraidos = {"error": f"Error al extraer datos de imagen: {str(e)}"}

    # ---------------------------------------------------------
    # EXTRACCIÓN PDF
    # ---------------------------------------------------------
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

    # --- EJECUTAR ANÁLISIS DE IA CON EL CONTENIDO CRUDO ---
    analisis_ia = evaluar_presencia_ia(metadatos_extraidos, contenido)

    return {
        "archivo": file.filename,
        "hash_sha256": hash_resultado,
        "evaluacion_ia": analisis_ia,
        "metadatos_completos": metadatos_extraidos
    }