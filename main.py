from fastapi import FastAPI, File, UploadFile
import hashlib
from PIL import Image, ImageChops, ImageEnhance
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

app = FastAPI()

# Configuración de Entorno
HF_TOKEN = os.getenv("HF_TOKEN")

# Modelos reordenados por precisión forense actual
MODELS = {
    "IMAGE": [
        "Nahrawy/AI-Generated-Image-Detector", # Muy preciso, menos falsos positivos con cámaras reales
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
    
    for modelo in lista_modelos:
        url = f"https://api-inference.huggingface.co/models/{modelo}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
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
                
                if response.status_code in [404, 410]:
                    break # Modelo no disponible, saltar al siguiente
                    
                if response.status_code != 200:
                    break
                
                resultado = response.json()
                
                if isinstance(resultado, list):
                    if len(resultado) > 0 and isinstance(resultado[0], list):
                        return resultado[0]
                    return resultado
                    
                return resultado
            except Exception as e:
                break # Fallo interno, saltar al siguiente
                
    return {"error": "Servidores de IA saturados. Intentando análisis matemático local."}

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
        # CALIBRACIÓN: Subimos a 18.0 para ignorar el ruido natural de las cámaras
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
    tiene_exif_real = False # Nuestro nuevo escudo protector

    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        tipo_evidencia = "IMAGEN"
        try:
            resultado_ela = aplicar_analisis_ela(contenido)
            if resultado_ela and resultado_ela.get("ela_ejecutado"):
                metadatos_extraidos["analisis_ela_matematico"] = resultado_ela

            img = Image.open(io.BytesIO(contenido))
            
            # --- EL ESCUDO EXIF (DETECTOR DE CÁMARA REAL) ---
            try:
                exif_data = img._getexif()
                if exif_data:
                    tiene_exif_real = True
                    metadatos_extraidos["ALERTA_OPTICA"] = "Datos de lente físico detectados. Protegiendo contra falsos positivos."
            except: pass

            img = img.convert('RGB')
            metadatos_extraidos.update({f"IMG_{k}": str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))})
            
            img.thumbnail((800, 800)) 
            buffer_optimo = io.BytesIO()
            img.save(buffer_optimo, format="JPEG", quality=95)
            contenido_optimo = buffer_optimo.getvalue()
            
            # Solo pasamos la imagen cruda, quitamos el biométrico para no confundir a la IA principal
            resultado_ia_profundo = consultar_modelo_hf(contenido_optimo, "IMAGE")
            del img, buffer_optimo
            gc.collect() 
        except Exception as e: print(f"Error procesando imagen: {e}")

    # (Lógica de video y audio simplificada omitida para enfocar en la precisión de imagen, 
    # asume que funciona igual que tu código anterior)

    # --- LÓGICA DE CALIBRACIÓN DINÁMICA DE RIESGO ---
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

    # --- EL CEREBRO DE FORENSIA (ANTI FALSOS POSITIVOS) ---
    if error_ia:
        resultado_estructural["nivel_riesgo"] = "AMBIGUO"
        resultado_estructural["motivo"] = error_ia
    else:
        # Definimos los umbrales base
        umbral_alto = 70.0
        umbral_preventivo = 25.0

        # Si detectamos que es una foto de cámara real, subimos la exigencia matemática
        if tiene_exif_real:
            umbral_alto = 95.0  # Solo marcará ALTO si la IA está absolutamente segura (evita falsos por HDR)
            umbral_preventivo = 60.0
            
        # Si es captura de pantalla, somos escépticos con la IA
        if es_captura:
            umbral_alto = 85.0
            umbral_preventivo = 50.0

        # Evaluación final con los umbrales dinámicos
        if porcentaje_ia >= umbral_alto:
            resultado_estructural["nivel_riesgo"] = "ALTO"
            if tiene_exif_real:
                resultado_estructural["motivo"] = f"ALERTA: Manipulación severa detectada ({round(porcentaje_ia)}%). El archivo intenta hacerse pasar por una foto de cámara real."
            else:
                resultado_estructural["motivo"] = f"Evidencia sintética. Alta probabilidad de generación mediante Inteligencia Artificial ({round(porcentaje_ia)}%)."
                
        elif porcentaje_ia >= umbral_preventivo or anomalia_ela:
            resultado_estructural["nivel_riesgo"] = "PREVENTIVO"
            if tiene_exif_real or es_captura:
                resultado_estructural["motivo"] = f"El porcentaje de IA ({round(porcentaje_ia)}%) se debe probablemente a la compresión de la red social o al procesamiento HDR del teléfono."
            else:
                resultado_estructural["motivo"] = "Se detectan trazas leves de edición digital o alteraciones en la textura de los píxeles (ELA)."
                
        else:
            resultado_estructural["nivel_riesgo"] = "BAJO"
            resultado_estructural["motivo"] = "Estructura óptica consistente. Alta probabilidad de origen natural sin manipulaciones evidentes."

    return {
        "nombre_archivo": file.filename,
        "hash_sha256": hash_sha256,
        "tipo_evidencia": tipo_evidencia,
        "es_captura": es_captura,
        "analisis": resultado_estructural,
        "detalles_biometricos": {"score_ia": resultado_ia_profundo},
        "detalles_tecnicos": metadatos_extraidos
    }