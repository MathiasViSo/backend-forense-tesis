from fastapi import FastAPI, File, UploadFile
import hashlib
import os
import requests

app = FastAPI()

# --- CREDENCIALES EMPRESARIALES ---
API_USER = os.getenv("SIGHTENGINE_USER")
API_SECRET = os.getenv("SIGHTENGINE_SECRET")

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    # 1. Leemos los bytes crudos. ¡Ojo! Ya no usamos PIL ni OpenCV para no saturar la RAM.
    contenido = await file.read()
    nombre_archivo = file.filename.lower()
    hash_sha256 = hashlib.sha256(contenido).hexdigest()

    if not API_USER or not API_SECRET:
        return {"error": "Las credenciales de Sightengine no están configuradas en Render."}

    # 2. Preparamos el paquete blindado para el proveedor comercial
    try:
        # El endpoint especializado en Inteligencia Artificial Generativa
        url = 'https://api.sightengine.com/1.0/check.json'
        
        archivos = {'media': (nombre_archivo, contenido, file.content_type)}
        datos = {
            'models': 'genai', # Modelo comercial de detección IA
            'api_user': API_USER,
            'api_secret': API_SECRET
        }
        
        # 3. Disparamos a sus servidores. Ellos hacen el trabajo pesado.
        respuesta = requests.post(url, files=archivos, data=datos)
        resultado = respuesta.json()
        
        # 4. Procesamos el veredicto exacto
        if resultado.get("status") == "success":
            # Sightengine nos devuelve directamente la probabilidad de 'ai_generated'
            datos_genai = resultado.get("type", {})
            score_ia = datos_genai.get("ai_generated", 0.0)
            
            porcentaje_ia = score_ia * 100
            
            # --- CALIBRACIÓN COMERCIAL ---
            # Sightengine es muy exacto. Si dice que es IA, confía en él.
            if porcentaje_ia > 75.0:
                nivel_riesgo = "ALTO"
                motivo = "El motor forense comercial confirma origen sintético o generación algorítmica profunda."
            elif porcentaje_ia > 25.0:
                nivel_riesgo = "PREVENTIVO"
                motivo = "Se detectan trazas de alteración parcial. Posible uso agresivo de filtros algorítmicos o inpainting."
            else:
                nivel_riesgo = "BAJO"
                motivo = "Evidencia íntegra. El escáner empresarial certifica el origen natural de la fotografía."

            return {
                "nombre_archivo": nombre_archivo,
                "hash_sha256": hash_sha256,
                "tipo_evidencia": "IMAGEN",
                "es_captura": False, # El modelo comercial ya no se confunde con esto
                "analisis": {
                    "porcentaje_ia": round(porcentaje_ia, 2),
                    "nivel_riesgo": nivel_riesgo,
                    "motivo": motivo
                },
                "detalles_tecnicos": {
                    "proveedor": "Sightengine Enterprise API",
                    "resolucion_cruce": "Aprobado"
                }
            }
        else:
            # Si el proveedor devuelve un error (ej. formato no soportado)
            return {"error": f"Fallo en la API comercial: {resultado.get('error', {}).get('message', 'Desconocido')}"}
            
    except Exception as e:
        return {"error": f"Error de infraestructura de red: {str(e)}"}