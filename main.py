from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS
from pypdf import PdfReader
import io
import requests
import os # <-- NUEVO: Importamos os

app = FastAPI() 

# --- CONFIGURACIÓN DE HUGGING FACE (Segura) ---
# Ahora lee la variable secreta desde el servidor de Render
TOKEN_HF = os.environ.get("HUGGINGFACE_TOKEN") 
API_URL = "https://api-inference.huggingface.co/models/dima806/ai_vs_real_image_detection"
headers = {"Authorization": f"Bearer {TOKEN_HF}"}


FIRMAS_IA = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", 
    "adobe firefly", "generative fill", "ai generated", "comfyui", "bing"
]

def evaluar_presencia_ia(metadatos: dict, contenido_crudo: bytes, es_imagen: bool) -> dict:
    """Análisis híbrido: 1ro Metadatos (Estático), 2do Píxeles (Red Neuronal)"""
    texto_evidencia = str(metadatos).lower()
    texto_binario = contenido_crudo.decode('utf-8', errors='ignore').lower()
    
    # 1. FILTRO ESTÁTICO (Metadatos y Bytes)
    for firma in FIRMAS_IA:
        if firma in texto_evidencia or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "metodo": "Análisis Estático (Firmas/Metadatos)",
                "motivo": f"Se encontró la firma '{firma.upper()}' incrustada en el archivo."
            }
            
    # 2. FILTRO NEURONAL (Píxeles)
    if es_imagen:
        try:
            # Enviamos a Hugging Face
            response = requests.post(API_URL, headers=headers, data=contenido_crudo)
            
            if response.status_code == 200:
                resultados = response.json()
                # Buscamos si detecta IA (ampliamos las palabras clave por si acaso)
                for res in resultados:
                    etiqueta = str(res.get("label", "")).lower()
                    if etiqueta in ["artificial", "fake", "ai", "generated"]:
                        probabilidad = res.get("score", 0) * 100
                        if probabilidad > 50:
                            return {
                                "detectado": True,
                                "nivel_riesgo": "ALTO",
                                "metodo": "Análisis de Píxeles (Hugging Face)",
                                "motivo": f"Red neuronal detectó '{etiqueta}' con {probabilidad:.2f}% de certeza."
                            }
                # Si respondió bien pero no detectó IA, mostramos qué fue lo que vio
                return {
                    "detectado": False,
                    "nivel_riesgo": "BAJO",
                    "metodo": "Análisis de Píxeles (Hugging Face)",
                    "motivo": f"Resultados crudos del modelo: {resultados}"
                }
                
            elif response.status_code == 503:
                return {
                    "detectado": False,
                    "nivel_riesgo": "DESCONOCIDO",
                    "metodo": "Hugging Face Cargando",
                    "motivo": "El servidor de IA está 'despertando'. Espera 30 segundos y vuelve a presionar Analizar."
                }
            else:
                return {
                    "detectado": False,
                    "nivel_riesgo": "ERROR",
                    "metodo": f"Fallo API (HTTP {response.status_code})",
                    "motivo": f"Respuesta de Hugging Face: {response.text}"
                }
        except Exception as e:
            return {
                "detectado": False,
                "nivel_riesgo": "ERROR",
                "metodo": "Fallo Interno",
                "motivo": f"Error de conexión: {str(e)}"
            }

    # Si no es imagen y pasó el filtro 1
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "metodo": "Análisis Estático",
        "motivo": "No es una imagen, y no se encontraron firmas en el texto."
    }

    """Análisis híbrido: 1ro Metadatos (Estático), 2do Píxeles (Red Neuronal)"""
    texto_evidencia = str(metadatos).lower()
    texto_binario = contenido_crudo.decode('utf-8', errors='ignore').lower()
    
    # 1. FILTRO ESTÁTICO (Metadatos y Bytes)
    for firma in FIRMAS_IA:
        if firma in texto_evidencia or firma in texto_binario:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "metodo": "Análisis Estático (Firmas/Metadatos)",
                "motivo": f"Se encontró la firma '{firma.upper()}' incrustada en el archivo."
            }
            
    # 2. FILTRO NEURONAL (Solo si es imagen y pasó el primer filtro)
    if es_imagen:
        es_ia, probabilidad = analizar_pixeles_ia(contenido_crudo)
        if es_ia:
            return {
                "detectado": True,
                "nivel_riesgo": "ALTO",
                "metodo": "Análisis de Píxeles (Visión Artificial)",
                "motivo": f"La red neuronal detectó patrones sintéticos con {probabilidad}% de certeza."
            }

    # Si pasa ambos filtros, está limpia
    return {
        "detectado": False,
        "nivel_riesgo": "BAJO",
        "metodo": "Análisis Híbrido Completo",
        "motivo": "No se encontraron rastros de IA en metadatos ni en análisis de píxeles."
    }

@app.get("/")
def home():
    return {"mensaje": "API Forense Híbrida (Metadatos + Red Neuronal) activa"}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    hash_resultado = hashlib.sha256(contenido).hexdigest()
    metadatos_extraidos = {}
    
    nombre_archivo = file.filename.lower() if file.filename else ""
    tipo_contenido = file.content_type if file.content_type else ""
    es_imagen = False
    
    # ---------------------------------------------------------
    # ANÁLISIS DE IMÁGENES
    # ---------------------------------------------------------
    if tipo_contenido.startswith("image/") or nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.jfif')):
        es_imagen = True
        try:
            imagen = Image.open(io.BytesIO(contenido))
            
            # A) Metadatos EXIF
            if hasattr(imagen, '_getexif') and imagen._getexif():
                for tag_id, valor in imagen._getexif().items():
                    nombre_tag = TAGS.get(tag_id, tag_id)
                    metadatos_extraidos[f"EXIF_{nombre_tag}"] = str(valor)
            
            # B) Metadatos PNG
            if hasattr(imagen, 'info') and imagen.info:
                for clave, valor in imagen.info.items():
                    if isinstance(valor, str) or isinstance(valor, bytes):
                        metadatos_extraidos[f"PNG_{clave}"] = str(valor)

            if not metadatos_extraidos:
                metadatos_extraidos = {"aviso": "Imagen sin metadatos legibles (posible lavado)."}
                
        except Exception as e:
            metadatos_extraidos = {"error": f"Error al leer la imagen: {str(e)}"}

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

    # --- EJECUTAR ANÁLISIS HÍBRIDO ---
    analisis_ia = evaluar_presencia_ia(metadatos_extraidos, contenido, es_imagen)

    return {
        "archivo": file.filename,
        "hash_sha256": hash_resultado,
        "evaluacion_ia": analisis_ia,
        "metadatos_completos": metadatos_extraidos
    }