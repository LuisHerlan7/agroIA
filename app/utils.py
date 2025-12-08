import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from pathlib import Path
import json
import io

MODEL_PATH = Path('../models/modelo_hojas.h5')
CLASS_INDICES_PATH = Path('../models/class_indices.json')
IMG_SIZE = (224, 224)

_model = None
_class_indices = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH)
    return _model

def load_class_indices():
    global _class_indices
    if _class_indices is None:
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(f"Índices de clases no encontrados en {CLASS_INDICES_PATH}")
        with open(CLASS_INDICES_PATH, 'r') as f:
            _class_indices = json.load(f)
    return _class_indices

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image_bytes):
    model = load_model()
    class_indices = load_class_indices()
    
    processed_img = preprocess_image(image_bytes)
    predictions = model.predict(processed_img, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence

