from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS
from pypdf import PdfReader # Importación para PDF
import io

# ESTA LÍNEA ES LA QUE FALTA O ESTÁ MAL UBICADA:
app = FastAPI() 

@app.get("/")
def home():
    return {"mensaje": "API Forense con Metadatos activa"}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    # ... todo el resto del código que pusimos antes ...
    contenido = await file.read()
    hash_resultado = hashlib.sha256(contenido).hexdigest()
    metadatos_extraidos = {}
    
    # SI ES IMAGEN
    if file.content_type.startswith("image/"):
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
            metadatos_extraidos = {"error": f"Error en imagen: {str(e)}"}

    # SI ES PDF
    elif file.content_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(contenido))
            info = reader.metadata
            if info:
                for key, value in info.items():
                    metadatos_extraidos[key.replace("/", "")] = str(value)
            else:
                metadatos_extraidos = {"aviso": "PDF sin metadatos internos"}
        except Exception as e:
            metadatos_extraidos = {"error": f"Error en PDF: {str(e)}"}
            
    else:
        metadatos_extraidos = {"aviso": "Tipo de archivo no soportado"}

    return {
        "archivo": file.filename,
        "hash_sha256": hash_resultado,
        "metadatos": metadatos_extraidos
    }