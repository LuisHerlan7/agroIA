import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from pathlib import Path
import json
import io

# Use absolute paths from the project root
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'modelo_hojas.h5'
CLASS_INDICES_PATH = PROJECT_ROOT / 'models' / 'class_indices.json'
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
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Imagen vacía o inválida")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        # Re-raise with a clearer message for the API response
        raise ValueError(f"No se pudo abrir la imagen: {e}")
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_disease_info():
    """Load disease information and treatment recommendations"""
    global _disease_info
    if not hasattr(load_disease_info, '_cache'):
        try:
            disease_info_path = PROJECT_ROOT / 'models' / 'disease_info.json'
            if disease_info_path.exists():
                with open(disease_info_path, 'r', encoding='utf-8') as f:
                    load_disease_info._cache = json.load(f)
            else:
                load_disease_info._cache = {}
        except Exception:
            load_disease_info._cache = {}
    return load_disease_info._cache

def predict_disease(image_bytes):
    model = load_model()
    class_indices = load_class_indices()
    disease_info = load_disease_info()
    
    processed_img = preprocess_image(image_bytes)
    predictions = model.predict(processed_img, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    # Get detailed info if available
    info = disease_info.get(predicted_class, {})
    
    return {
        "disease_key": predicted_class,
        "disease_name": info.get("nombre_display", predicted_class),
        "description": info.get("descripcion", ""),
        "treatment": info.get("tratamiento", ""),
        "confidence": confidence
    }

