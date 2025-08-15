from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import shutil
import uuid

app = Flask(__name__)

# Directories for storing images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DETECTED_FOLDER = os.path.join('static', 'detected')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# ✅ Path to your updated .keras model
MODEL_PATH = 'resaved_model.keras'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the model
model = load_model(MODEL_PATH)

# Labels based on training
class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save file to uploads
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Copy image to detected
        detected_path = os.path.join(DETECTED_FOLDER, filename)
        shutil.copy(filepath, detected_path)

        return render_template('result.html',
                               image_path=filepath,
                               prediction=predicted_class,
                               confidence=f"{confidence:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
