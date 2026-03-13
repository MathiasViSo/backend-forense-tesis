from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS
from pypdf import PdfReader
import io

app = FastAPI() 

@app.get("/")
def home():
    return {"mensaje": "API Forense con Metadatos activa"}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    hash_resultado = hashlib.sha256(contenido).hexdigest()
    metadatos_extraidos = {}
    
    # Obtenemos el nombre en minúsculas y el tipo de contenido de forma segura
    nombre_archivo = file.filename.lower() if file.filename else ""
    tipo_contenido = file.content_type if file.content_type else ""
    
    # ---------------------------------------------------------
    # 1. ANÁLISIS DE IMÁGENES
    # ---------------------------------------------------------
    # Si la cabecera dice que es imagen, o si termina en jpg/png/jpeg
    if tipo_contenido.startswith("image/") or nombre_archivo.endswith(('.png', '.jpg', '.jpeg')):
        try:
            imagen = Image.open(io.BytesIO(contenido))
            info_exif = imagen._getexif()
            if info_exif:
                for tag_id, valor in info_exif.items():
                    nombre_tag = TAGS.get(tag_id, tag_id)
                    metadatos_extraidos[nombre_tag] = str(valor)
            else:
                metadatos_extraidos = {"aviso": "Imagen sin metadatos EXIF (posiblemente limpia por red social)"}
        except Exception as e:
            metadatos_extraidos = {"error": f"Error al extraer datos de imagen: {str(e)}"}

    # ---------------------------------------------------------
    # 2. ANÁLISIS DE PDF
    # ---------------------------------------------------------
    # Si la cabecera dice que es PDF, o si la extensión es .pdf
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
            metadatos_extraidos = {"error": f"Error al extraer datos del PDF: {str(e)}"}
            
    # ---------------------------------------------------------
    # 3. ARCHIVO NO RECONOCIDO
    # ---------------------------------------------------------
    else:
        metadatos_extraidos = {
            "aviso": "Tipo de archivo no soportado para análisis profundo",
            "detalles_tecnicos": f"El móvil lo envió como: {tipo_contenido}"
        }

    return {
        "archivo": file.filename,
        "hash_sha256": hash_resultado,
        "metadatos": metadatos_extraidos
    }