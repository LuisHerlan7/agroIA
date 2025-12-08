#!/usr/bin/env python
"""Generate a minimal dummy model for testing"""
import json
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

print("Creando un modelo dummy para testing...")

# Create a minimal model
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')  # 4 clases de ejemplo
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model_path = MODEL_DIR / 'modelo_hojas.h5'
model.save(model_path)
print(f"✓ Modelo guardado en: {model_path}")

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
print(f"✓ Class indices guardados en: {indices_path}")

# Verify
print("\nVerificando...")
print(f"Tamaño del modelo: {model_path.stat().st_size} bytes")
print(f"Clases: {json.dumps(class_indices, indent=2)}")
print("\n✓ ¡Listo! Ahora puedes usar la app para testing.")
