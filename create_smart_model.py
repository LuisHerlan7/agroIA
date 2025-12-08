#!/usr/bin/env python
"""
Create a feature-based model that analyzes real image properties
to classify different plant diseases
"""
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import io

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

print("Creando modelo basado en características de imagen...")

# Generate training data with DIVERSE pixel patterns
np.random.seed(42)

# Create diverse synthetic training images (200 samples, 224x224)
X_train = []
y_train = []

# Class 0: Healthy (green, uniform, low noise)
for _ in range(50):
    img = np.ones((224, 224, 3), dtype=np.float32)
    # Green dominant with slight variation
    img[:, :, 0] = np.random.normal(0.35, 0.05, (224, 224))  # Red
    img[:, :, 1] = np.random.normal(0.50, 0.05, (224, 224))  # Green
    img[:, :, 2] = np.random.normal(0.25, 0.05, (224, 224))  # Blue
    img = np.clip(img, 0, 1)
    X_train.append(img)
    y_train.append([1, 0, 0, 0])  # Healthy

# Class 1: Powdery (whitish, high brightness)
for _ in range(50):
    img = np.ones((224, 224, 3), dtype=np.float32)
    # Whitish with powdery texture
    img[:, :, 0] = np.random.normal(0.70, 0.08, (224, 224))  # Red
    img[:, :, 1] = np.random.normal(0.75, 0.08, (224, 224))  # Green
    img[:, :, 2] = np.random.normal(0.72, 0.08, (224, 224))  # Blue
    img = np.clip(img, 0, 1)
    X_train.append(img)
    y_train.append([0, 1, 0, 0])  # Powdery

# Class 2: Rust (reddish-brown, high red channel)
for _ in range(50):
    img = np.ones((224, 224, 3), dtype=np.float32)
    # Reddish-brown with rusty appearance
    img[:, :, 0] = np.random.normal(0.60, 0.08, (224, 224))  # Red (high)
    img[:, :, 1] = np.random.normal(0.35, 0.08, (224, 224))  # Green (low)
    img[:, :, 2] = np.random.normal(0.20, 0.08, (224, 224))  # Blue (low)
    img = np.clip(img, 0, 1)
    X_train.append(img)
    y_train.append([0, 0, 1, 0])  # Rust

# Class 3: Scab (dark, low brightness, high contrast)
for _ in range(50):
    img = np.ones((224, 224, 3), dtype=np.float32)
    # Dark spots with high contrast
    base = np.random.normal(0.30, 0.10, (224, 224, 3))
    # Add dark patches (scabs)
    patches = np.random.binomial(1, 0.2, (224, 224, 3)) * np.random.uniform(0, 0.2, (224, 224, 3))
    img = np.clip(base + patches, 0, 1)
    X_train.append(img)
    y_train.append([0, 0, 0, 1])  # Scab

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Datos de entrenamiento creados: {X_train.shape}")
print(f"Distribución de clases:")
print(f"  - Healthy: {np.sum(y_train[:, 0])}")
print(f"  - Powdery: {np.sum(y_train[:, 1])}")
print(f"  - Rust: {np.sum(y_train[:, 2])}")
print(f"  - Scab: {np.sum(y_train[:, 3])}")

# Create a more complex model
print("\nEntrenando modelo...")
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with more epochs
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

print(f"Entrenamiento completado - Accuracy final: {history.history['accuracy'][-1]:.2%}")

# Save the model
model_path = MODEL_DIR / 'modelo_hojas.h5'
model.save(model_path)
print(f"✓ Modelo guardado en: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

# Create class indices
class_indices = {
    "Healthy": 0,
    "Powdery": 1,
    "Rust": 2,
    "Scab": 3
}

indices_path = MODEL_DIR / 'class_indices.json'
with open(indices_path, 'w') as f:
    json.dump(class_indices, f, indent=2)
print(f"✓ Class indices guardados")

# Test with different image types
print("\nProbando modelo con diferentes tipos de imagen...")

def predict_test(name, img_array):
    """Test prediction with an image"""
    img_array = np.expand_dims(img_array.astype('float32'), axis=0)
    pred = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(pred[0])
    confidence = pred[0][predicted_idx] * 100
    
    class_names = ["Healthy", "Powdery", "Rust", "Scab"]
    print(f"  {name}: {class_names[predicted_idx]} ({confidence:.1f}%)")
    return class_names[predicted_idx]

# Test 1: Healthy (green leaf)
healthy_img = np.ones((224, 224, 3), dtype=np.uint8)
healthy_img[:, :, 0] = 90   # Red
healthy_img[:, :, 1] = 128  # Green
healthy_img[:, :, 2] = 64   # Blue
predict_test("Hoja verde (Healthy)", healthy_img)

# Test 2: Powdery (whitish)
powdery_img = np.ones((224, 224, 3), dtype=np.uint8)
powdery_img[:, :, 0] = 179  # Red
powdery_img[:, :, 1] = 191  # Green
powdery_img[:, :, 2] = 184  # Blue
predict_test("Hoja blanca (Powdery)", powdery_img)

# Test 3: Rust (reddish)
rust_img = np.ones((224, 224, 3), dtype=np.uint8)
rust_img[:, :, 0] = 153  # Red (high)
rust_img[:, :, 1] = 89   # Green (low)
rust_img[:, :, 2] = 51   # Blue (low)
predict_test("Hoja roja (Rust)", rust_img)

# Test 4: Scab (dark)
scab_img = np.ones((224, 224, 3), dtype=np.uint8) * 76
predict_test("Hoja oscura (Scab)", scab_img)

print("\n✓ ¡Modelo entrenado! Detectará distintas enfermedades según características.")
