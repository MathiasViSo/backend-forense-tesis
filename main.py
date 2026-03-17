from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image, ImageChops, ImageEnhance, ImageFile
from pypdf import PdfReader
from mutagen import File as MutagenFile
import io
import os
import requests
import cv2
import numpy as np
import tempfile
import base64
import time
import gc

# Esto evita que PIL arroje error si la imagen subida desde el celular está ligeramente corrupta
ImageFile.LOAD_TRUNCATED_IMAGES = True 

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS = {
    "IMAGE": [
        "Nahrawy/AI-Generated-Image-Detector",
        "dima806/ai_vs_real_image_detection", 
        "umm-maybe/AI-image-detector"
    ], 
    "AUDIO": [
        "ResembleAI/ai_detector_audio"
    ]
}

FIRMAS_IA = [
    "midjourney", "dall-e", "stable diffusion", "adobe firefly", 
    "generative fill", "ai generated", "comfyui", "synthid", 
    "lyria", "veo", "gemini", "deepfake", "roop", "bing"
]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def consultar_modelo_hf(contenido_bytes: bytes, tipo: str, max_intentos_por_modelo=3):
    if not HF_TOKEN or tipo not in MODELS:
        return {"error": "Credenciales de IA no configuradas en el servidor."}
    
    lista_modelos = MODELS[tipo]
    ultimo_error = "Desconocido"
    tipo_mime = "image/jpeg" if tipo == "IMAGE" else "application/octet-stream"
    
    for modelo in lista_modelos:
        url = f"https://router.huggingface.co/hf-inference/models/{modelo}"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": tipo_mime
        }
        
        for intento in range(max_intentos_por_modelo):
            try:
                response = requests.post(url, headers=headers, data=contenido_bytes)
                
                if response.status_code == 503:
                    try:
                        t = float(response.json().get("estimated_time", 15.0))
                        time.sleep(t + 2)
                        continue
                    except:
                        time.sleep(10)
                        continue
                
                if response.status_code != 200:
                    ultimo_error = f"HTTP {response.status_code} en {modelo}: {response.text}"
                    break 
                
                resultado = response.json()
                
                if isinstance(resultado, list):
                    if len(resultado) > 0 and isinstance(resultado[0], list):
                        return resultado[0]
                    return resultado
                    
                return resultado
            except Exception as e:
                ultimo_error = f"Fallo interno de red: {str(e)}"
                break
                
    return {"error": f"Detalle técnico del bloqueo: {ultimo_error}"}

def aplicar_analisis_ela(contenido_imagen: bytes, calidad_recompresion: int = 90) -> dict:
    try:
        imagen_original = Image.open(io.BytesIO(contenido_imagen)).convert('RGB')
        buffer_temporal = io.BytesIO()
        imagen_original.save(buffer_temporal, 'JPEG', quality=calidad_recompresion)
        buffer_temporal.seek(0)
        imagen_recomprimida = Image.open(buffer_temporal)
        
        diferencia_ela = ImageChops.difference(imagen_original, imagen_recomprimida)
        ela_array = np.array(diferencia_ela)
        
        desviacion_ruido = float(np.std(ela_array))
        sospecha_manipulacion = desviacion_ruido > 18.0 
        
        extremos = diferencia_ela.getextrema()
        max_diff = max([ex[1] for ex in extremos])
        if max_diff == 0: max_diff = 1
        escala_brillo = 255.0 / max_diff
        imagen_ela_resaltada = ImageEnhance.Brightness(diferencia_ela).enhance(escala_brillo)
        
        buffer_salida = io.BytesIO()
        imagen_ela_resaltada.save(buffer_salida, format="JPEG")
        mapa_calor_b64 = base64.b64encode(buffer_salida.getvalue()).decode("utf-8")
        
        return {
            "ela_ejecutado": True,
            "desviacion_estandar_ruido": round(desviacion_ruido, 2),
            "anomalia_detectada_ela": sospecha_manipulacion,
            "mapa_calor_base64": mapa_calor_b64
        }
    except Exception as e:
        return {"ela_ejecutado": False, "error_ela": str(e)}

def realizar_analisis_forense_metadatos(contenido: bytes, metadatos: dict, es_captura: bool) -> dict:
    texto_metadatos = str(metadatos).lower()
    texto_binario = contenido.decode('utf-8', errors='ignore').lower()
    for firma in FIRMAS_IA:
        if firma in texto_metadatos or firma in texto_binario:
            return {"detectado": True, "nivel_riesgo": "ALTO", "motivo": f"Firma generativa '{firma.upper()}' detectada en el código base."}
    if es_captura:
        return {"detectado": False, "nivel_riesgo": "PREVENTIVO", "motivo": "Metadatos purgados (Captura/Red Social)."}
    return {"detectado": False, "nivel_riesgo": "BAJO", "motivo": "Sin firmas conocidas."}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()
    
    palabras_captura = ['screenshot', 'captura', 'whatsapp', 'screen_', 'image-', 'img-', 'instagram']
    es_captura = any(p in nombre_archivo for p in palabras_captura)
    
    metadatos_extraidos = {}
    resultado_ia_profundo = None
    resultado_ela = None
    tipo_evidencia = "DESCONOCIDO"
    tiene_exif_real = False 

    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGEN"
        try:
            # 1. ESCUDO DE MEMORIA RAM PARA RENDER (Evita colapsos con fotos de >10MB)
            img_original = Image.open(io.BytesIO(contenido))
            
            try:
                exif_data = img_original._getexif()
                if exif_data:
                    tiene_exif_real = True
                    metadatos_extraidos["ALERTA_OPTICA"] = "Datos de lente físico detectados."
            except: pass

            # Si la foto es un monstruo gigante, la achicamos manteniendo su proporción
            if img_original.width > 2000 or img_original.height > 2000:
                # thumbnail modifica la imagen "in place"
                img_original.thumbnail((2000, 2000))
                
            img_original = img_original.convert('RGB')
            metadatos_extraidos.update({f"IMG_{k}": str(v) for k, v in img_original.info.items() if isinstance(v, (str, bytes))})
            
            # Guardamos la imagen procesada y segura en memoria
            buffer_seguro = io.BytesIO()
            img_original.save(buffer_seguro, format="JPEG", quality=98)
            contenido_seguro = buffer_seguro.getvalue()

            # 2. Análisis ELA con la imagen segura (no colapsará)
            resultado_ela = aplicar_analisis_ela(contenido_seguro)
            if resultado_ela and resultado_ela.get("ela_ejecutado"):
                metadatos_extraidos["analisis_ela_matematico"] = resultado_ela

            # 3. Preparamos una versión ultraligera solo para la IA
            img_ia = img_original.copy()
            img_ia.thumbnail((800, 800)) 
            buffer_ia = io.BytesIO()
            img_ia.save(buffer_ia, format="JPEG", quality=95)
            
            resultado_ia_profundo = consultar_modelo_hf(buffer_ia.getvalue(), "IMAGE")
            
            # Limpieza profunda de RAM
            del img_original, img_ia, buffer_seguro, buffer_ia
            gc.collect() 
        except Exception as e: print(f"Error procesando imagen: {e}")

    # (Lógica omitida de video/audio, puedes integrar tu código previo si los necesitas)

    # --- CEREBRO FORENSE DE CALIBRACIÓN DINÁMICA ---
    resultado_estructural = realizar_analisis_forense_metadatos(contenido, metadatos_extraidos, es_captura)
    porcentaje_ia = 0.0
    error_ia = None

    if isinstance(resultado_ia_profundo, list) and len(resultado_ia_profundo) > 0:
        score_data = sorted(resultado_ia_profundo, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top_label = str(score_data[0].get("label", "")).lower()
        top_score = float(score_data[0].get("score", 0.0))
        
        if any(x in top_label for x in ["human", "hum", "real", "genuine", "0"]):
            porcentaje_ia = (1.0 - top_score) * 100
        else:
            porcentaje_ia = top_score * 100
            
    elif isinstance(resultado_ia_profundo, dict) and "error" in resultado_ia_profundo:
        error_ia = resultado_ia_profundo["error"]

    anomalia_ela = resultado_ela.get("anomalia_detectada_ela", False) if resultado_ela else False
    resultado_estructural["porcentaje_ia"] = round(porcentaje_ia, 2)

    if error_ia:
        resultado_estructural["nivel_riesgo"] = "AMBIGUO"
        resultado_estructural["motivo"] = error_ia
    else:
        # --- EL MODO ESCÉPTICO ---
        if tiene_exif_real:
            # Foto de cámara pura. Confiamos en los umbrales normales.
            umbral_alto = 75.0
            umbral_preventivo = 35.0
        else:
            # Captura de pantalla, WhatsApp o sin metadatos.
            # Sabemos que la IA sufre alucinaciones aquí, así que somos súper exigentes.
            umbral_alto = 92.0
            umbral_preventivo = 65.0

        if porcentaje_ia >= umbral_alto:
            resultado_estructural["nivel_riesgo"] = "ALTO"
            if tiene_exif_real:
                resultado_estructural["motivo"] = f"ALERTA: Evidencia fotográfica sintética o severamente manipulada ({round(porcentaje_ia)}%)."
            else:
                resultado_estructural["motivo"] = f"A pesar de la compresión, existe altísima probabilidad de generación mediante IA ({round(porcentaje_ia)}%)."
                
        elif porcentaje_ia >= umbral_preventivo or anomalia_ela:
            resultado_estructural["nivel_riesgo"] = "PREVENTIVO"
            if not tiene_exif_real:
                resultado_estructural["motivo"] = f"Atención: El porcentaje ({round(porcentaje_ia)}%) se debe muy probablemente a la compresión de la captura o filtro de red social. Evidencia no concluyente."
            else:
                resultado_estructural["motivo"] = "Se detectan trazas leves de edición digital. Requiere revisión humana."
                
        else:
            resultado_estructural["nivel_riesgo"] = "BAJO"
            resultado_estructural["motivo"] = "Estructura óptica consistente. Alta probabilidad de origen natural sin manipulaciones evidentes."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "es_captura": es_captura or not tiene_exif_real,
        "analisis": resultado_estructural,
        "detalles_biometricos": {"score_ia": resultado_ia_profundo},
        "detalles_tecnicos": metadatos_extraidos
    }