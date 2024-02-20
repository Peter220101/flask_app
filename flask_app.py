from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import cv2
from keras.models import model_from_json

app = Flask(__name__)

# Rutas de los archivos del modelo y la imagen
model_h5 = "model.h5"
model_json = "model.json"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha enviado ninguna imagen'})

    image_file = request.files['image']

    # Guardamos la imagen en el servidor
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Cargamos tu modelo previamente entrenado
    with open(model_json, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5)

    # Cargamos la imagen
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (32, 32))
    image_normalized = np.float32(image_resized) / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    # Realizamos la predicción
    raw_result = model.predict(image_input)
    label_index = np.argmax(raw_result)
    labels = ['car', 'motorcycle', 'plane', 'train']
    prediction = labels[label_index]

    # Eliminamos la imagen temporal
    os.remove(image_path)

    # Devolvemos la predicción y la URL de la imagen cargada
    return jsonify({'prediction': prediction, 'image_url': f"/{image_path}"})


if __name__ == '__main__':
    app.run(debug=True)
