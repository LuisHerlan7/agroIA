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
NORM_PATH = PROJECT_ROOT / 'models' / 'normalization.json'
IMG_SIZE = (224, 224)

_model = None
_class_indices = None
_norm_params = None
_disease_info = None

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

def load_norm_params():
    """Load normalization parameters"""
    global _norm_params
    if _norm_params is None:
        if not NORM_PATH.exists():
            raise FileNotFoundError(f"Parámetros de normalización no encontrados en {NORM_PATH}")
        with open(NORM_PATH, 'r') as f:
            _norm_params = json.load(f)
    return _norm_params

def preprocess_image(image_bytes):
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Imagen vacía o inválida")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")
    
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    
    return img_array

def extract_features(image_array):
    """
    Extract color features from image array
    Returns: [mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast]
    """
    # Normalize to [0, 1]
    if image_array.max() > 1.0:
        image_array = image_array.astype('float32') / 255.0
    
    # Calculate channel statistics
    mean_r = float(np.mean(image_array[:, :, 0]))
    mean_g = float(np.mean(image_array[:, :, 1]))
    mean_b = float(np.mean(image_array[:, :, 2]))
    
    std_r = float(np.std(image_array[:, :, 0]))
    std_g = float(np.std(image_array[:, :, 1]))
    std_b = float(np.std(image_array[:, :, 2]))
    
    brightness = (mean_r + mean_g + mean_b) / 3.0
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])

def load_disease_info():
    """Load disease information and treatment recommendations"""
    global _disease_info
    if _disease_info is None:
        try:
            disease_info_path = PROJECT_ROOT / 'models' / 'disease_info.json'
            if disease_info_path.exists():
                with open(disease_info_path, 'r', encoding='utf-8') as f:
                    _disease_info = json.load(f)
            else:
                _disease_info = {}
        except Exception:
            _disease_info = {}
    return _disease_info

def predict_disease(image_bytes):
    model = load_model()
    class_indices = load_class_indices()
    norm_params = load_norm_params()
    disease_info = load_disease_info()
    
    # Preprocess image
    img_array = preprocess_image(image_bytes)
    
    # Extract features from the image
    features = extract_features(img_array)
    
    # Normalize features using stored parameters
    X_min = np.array(norm_params['X_min'])
    X_max = np.array(norm_params['X_max'])
    features_norm = (features - X_min) / (X_max - X_min + 1e-8)
    
    # Predict
    predictions = model.predict(features_norm.reshape(1, -1), verbose=0)
    
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
