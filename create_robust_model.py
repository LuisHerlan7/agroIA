#!/usr/bin/env python
"""
Create a direct rule-based model with clear thresholds for disease classification
This ensures consistent and interpretable results
"""
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

print("Creando modelo robusto basado en reglas de características...")

# Create a model that learns clear decision boundaries
np.random.seed(42)

X_train = []
y_train = []

# Generate clean training data with clear separation
n_samples_per_class = 100

# Class 0: Healthy - domina verde, valores medios
for _ in range(n_samples_per_class):
    mean_r = np.random.uniform(0.25, 0.35)   # Rojo bajo
    mean_g = np.random.uniform(0.45, 0.55)   # Verde alto
    mean_b = np.random.uniform(0.15, 0.25)   # Azul bajo
    std_r = np.random.uniform(0.04, 0.12)
    std_g = np.random.uniform(0.04, 0.12)
    std_b = np.random.uniform(0.04, 0.12)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([1, 0, 0, 0])

# Class 1: Powdery - todo claro/blanco
for _ in range(n_samples_per_class):
    mean_r = np.random.uniform(0.60, 0.75)   # Rojo alto
    mean_g = np.random.uniform(0.65, 0.80)   # Verde alto
    mean_b = np.random.uniform(0.60, 0.75)   # Azul alto
    std_r = np.random.uniform(0.01, 0.08)
    std_g = np.random.uniform(0.01, 0.08)
    std_b = np.random.uniform(0.01, 0.08)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 1, 0, 0])

# Class 2: Rust - rojo dominante, azul bajo
for _ in range(n_samples_per_class):
    mean_r = np.random.uniform(0.50, 0.65)   # Rojo alto
    mean_g = np.random.uniform(0.25, 0.40)   # Verde bajo
    mean_b = np.random.uniform(0.10, 0.25)   # Azul muy bajo
    std_r = np.random.uniform(0.05, 0.12)
    std_g = np.random.uniform(0.05, 0.12)
    std_b = np.random.uniform(0.04, 0.10)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 0, 1, 0])

# Class 3: Scab - oscuro, alto contraste
for _ in range(n_samples_per_class):
    mean_r = np.random.uniform(0.15, 0.30)   # Rojo bajo
    mean_g = np.random.uniform(0.12, 0.28)   # Verde bajo
    mean_b = np.random.uniform(0.10, 0.25)   # Azul bajo
    std_r = np.random.uniform(0.12, 0.20)    # Contraste alto
    std_g = np.random.uniform(0.12, 0.20)
    std_b = np.random.uniform(0.10, 0.18)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 0, 0, 1])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Distribución:")
print(f"  - Healthy: {np.sum(y_train[:, 0])}")
print(f"  - Powdery: {np.sum(y_train[:, 1])}")
print(f"  - Rust: {np.sum(y_train[:, 2])}")
print(f"  - Scab: {np.sum(y_train[:, 3])}")

# Normalize using simple min-max on training data (will be more robust)
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)

print("\nEntrenando modelo...")

model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"Entrenamiento completado - Accuracy final: {history.history['accuracy'][-1]:.2%}")

# Save model and normalization params
model_path = MODEL_DIR / 'modelo_hojas.h5'
model.save(model_path)
print(f"✓ Modelo guardado")

# Save normalization parameters
norm_params = {
    'X_min': X_min.tolist(),
    'X_max': X_max.tolist()
}

norm_path = MODEL_DIR / 'normalization.json'
with open(norm_path, 'w') as f:
    json.dump(norm_params, f)
print(f"✓ Parámetros de normalización guardados")

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
print(f"✓ Índices de clase guardados")

# Test with clear examples
print("\nProbando modelo...")

def test_predict(name, features):
    features_arr = np.array(features).reshape(1, -1)
    features_norm = (features_arr - X_min) / (X_max - X_min + 1e-8)
    pred = model.predict(features_norm, verbose=0)
    idx = np.argmax(pred[0])
    conf = pred[0][idx] * 100
    class_names = ["Healthy", "Powdery", "Rust", "Scab"]
    print(f"  {name}: {class_names[idx]} ({conf:.1f}%)")

# Healthy: green
test_predict("Verde", [0.30, 0.50, 0.20, 0.08, 0.08, 0.08, 0.33, 0.09])

# Powdery: white
test_predict("Blanco", [0.67, 0.72, 0.67, 0.03, 0.03, 0.03, 0.69, 0.03])

# Rust: red
test_predict("Rojo", [0.57, 0.32, 0.17, 0.08, 0.08, 0.07, 0.35, 0.08])

# Scab: dark
test_predict("Oscuro", [0.22, 0.20, 0.17, 0.16, 0.16, 0.14, 0.20, 0.15])

print("\n✓ ¡Modelo listo y probado!")
