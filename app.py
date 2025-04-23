from flask import Flask, request, jsonify, render_template
from predict import predict_image
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # This loads the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    result = predict_image(image_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
