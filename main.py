from flask import Flask, request, jsonify
import numpy as np
import cv2
import tempfile
from keras.models import load_model
import os
import logging

#---------

#---------

# === Configuración del log ===
logging.basicConfig(level=logging.DEBUG)

# === Carga del modelo y clasificador de rostro ===
model = load_model('modelo_emociones_25.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró archivo con clave 'file'"}), 400

    file = request.files['file']
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

# === Iniciar el servidor local ===
if __name__ == '__main__':
    app.run(port=5000, debug=True)
