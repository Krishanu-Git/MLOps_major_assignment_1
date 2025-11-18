from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = joblib.load('savedmodel.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Preprocess Image: Resize to 64x64, Grayscale
        img = Image.open(file).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).reshape(1, -1) # Flatten

        # Normalize if training data was normalized (Olivetti is usually 0-1 float)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        return f"Predicted Class (Person ID): {prediction[0]}"
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)