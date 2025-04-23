import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Constants
IMG_SIZE = (128, 128)
THRESHOLD = 0.6
class_names = ['Spiral', 'Elliptical']

# Load trained model
model = tf.keras.models.load_model("galaxy_model.keras")

def predict_image(file_storage, threshold=THRESHOLD):
    
    # Read file and convert to image
    image_bytes = file_storage.read()
    image_stream = io.BytesIO(image_bytes)

    img = tf.keras.utils.load_img(image_stream, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) 
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)[0] 
    
    # Predict
    predictions = model.predict(img_array)[0]

    # Top prediction
    top_index = np.argmax(predictions)
    top_confidence = predictions[top_index]
    predicted_class = class_names[top_index]

    # Threshold check for irregular
    if top_confidence < threshold:
        predicted_class = 'Irregular'

    print(f"Prediction: {predicted_class} (Confidence: {top_confidence:.2f})")

    return {
        'class_index': int(top_index),
        'class_name': predicted_class,
        'probability': float(top_confidence)
    }
