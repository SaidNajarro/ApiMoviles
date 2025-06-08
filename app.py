from flask import Flask, request, jsonify
import numpy as np
import cv2
import tempfile
from keras.models import load_model
import os
import logging
import requests

# === Configuración del log ===
logging.basicConfig(level=logging.DEBUG)

# === Descargar modelo si no existe ===
MODELO_ID = "1quWZTBuNOpoYi_YF0y3xVIuTI0MHqcu5"
MODELO_PATH = "modelo_emociones_25.keras"

def descargar_modelo_desde_gdrive(file_id, destination):
    """Descarga archivo de Google Drive manejando archivos grandes"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def descargar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("Descargando el modelo desde Google Drive...")
        try:
            descargar_modelo_desde_gdrive(MODELO_ID, MODELO_PATH)
            
            tamaño = os.path.getsize(MODELO_PATH)
            print(f"Modelo descargado. Tamaño: {tamaño} bytes")
            
            # Verificar que el archivo es válido (debe ser mayor a 100KB para un modelo)
            if tamaño < 100000:  # 100KB
                print(f"ADVERTENCIA: El archivo descargado es muy pequeño ({tamaño} bytes)")
                print("Esto podría indicar que se descargó una página HTML en lugar del modelo")
                
                # Leer las primeras líneas para verificar
                with open(MODELO_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                    primeras_lineas = f.read(500)
                    if 'html' in primeras_lineas.lower() or 'DOCTYPE' in primeras_lineas:
                        print("ERROR: Se descargó HTML en lugar del modelo")
                        os.remove(MODELO_PATH)
                        return False
            
            return True
            
        except Exception as e:
            print(f"Error descargando el modelo: {e}")
            if os.path.exists(MODELO_PATH):
                os.remove(MODELO_PATH)
            return False
    else:
        print(f"Modelo ya existe. Tamaño: {os.path.getsize(MODELO_PATH)} bytes")
    return True

# Intentar descargar el modelo
if not descargar_modelo():
    print("ADVERTENCIA: No se pudo descargar el modelo. La aplicación puede fallar.")
else:
    print("Modelo disponible para carga.")

# === Carga del modelo y clasificador de rostro ===
def cargar_modelo():
    try:
        print("Cargando modelo de emociones...")
        model = load_model(MODELO_PATH)
        print("Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

def cargar_clasificador():
    try:
        print("Cargando clasificador de rostros...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise Exception("No se pudo cargar el clasificador de rostros")
        print("Clasificador de rostros cargado exitosamente")
        return face_cascade
    except Exception as e:
        print(f"Error cargando clasificador: {e}")
        return None

# Cargar modelo y clasificador
model = cargar_modelo()
face_cascade = cargar_clasificador()

if model is None or face_cascade is None:
    print("ADVERTENCIA: No se pudieron cargar todos los componentes necesarios")

emotion_dict = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Surprise",
    5: "Sad",
    6: "Neutral"
}

# === Crear la aplicación Flask ===
app = Flask(__name__)

@app.route('/', methods=['POST'])
def detectar_emociones():
    # Verificar que los componentes estén cargados
    if model is None:
        return jsonify({"error": "Modelo no disponible"}), 503
    if face_cascade is None:
        return jsonify({"error": "Clasificador de rostros no disponible"}), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró archivo con clave 'file'"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó archivo"}), 400
        
    filename = file.filename.lower()

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file.read())
        temp_path = temp.name

    logging.debug(f"Archivo temporal guardado en: {temp_path}")

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            result = procesar_imagen(temp_path)
            os.unlink(temp_path)
            return jsonify({"tipo": "imagen", "resultados": result})

        elif filename.endswith(('.mp4', '.avi', '.mov')):
            result = procesar_video(temp_path)
            os.unlink(temp_path)
            return jsonify({"tipo": "video", "resultados": result})

        else:
            os.unlink(temp_path)
            return jsonify({"error": "Formato no soportado"}), 400

    except Exception as e:
        logging.error(f"Error procesando archivo: {e}", exc_info=True)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": "Error interno en el servidor", "detalle": str(e)}), 500

def procesar_imagen(path):
    frame = cv2.imread(path)
    if frame is None:
        return {"error": "No se pudo leer la imagen"}
    return analizar_frame(frame)

def procesar_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video"}

    emociones_totales = {emocion: 0 for emocion in emotion_dict.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        resultados_frame = analizar_frame(frame)
        for rostro in resultados_frame:
            emocion_principal = rostro.get("emocion_principal")
            if emocion_principal in emociones_totales:
                emociones_totales[emocion_principal] += 1

    cap.release()
    return emociones_totales

def analizar_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    resultados = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            roi = cv2.resize(roi, (48, 48))
        except Exception as e:
            logging.warning(f"No se pudo redimensionar ROI: {e}")
            continue

        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        pred = model.predict(roi)[0]
        emociones = {emotion_dict[i]: float(round(pred[i], 4)) for i in range(len(pred))}
        emocion_max = max(emociones, key=emociones.get)

        resultados.append({
            "coordenadas": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emociones": emociones,
            "emocion_principal": emocion_max
        })

    return resultados

# === Health check endpoint ===
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "modelo_cargado": model is not None,
        "clasificador_cargado": face_cascade is not None,
        "modelo_existe": os.path.exists(MODELO_PATH)
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensaje": "API de Detección de Emociones",
        "endpoints": {
            "/": "POST - Enviar archivo de imagen o video",
            "/health": "GET - Estado de la API"
        }
    })

# === Iniciar el servidor ===
if __name__ == '__main__':
    app.run(port=5000, debug=True)
