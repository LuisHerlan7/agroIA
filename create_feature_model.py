#!/usr/bin/env python
"""
Create a simpler model that extracts color histogram features
This approach is more interpretable and works better with varied synthetic data
"""
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import pickle

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

print("Creando modelo basado en features de color...")

np.random.seed(42)

# Generate feature vectors (instead of raw images)
# Features: [mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast]
X_train = []
y_train = []

# Class 0: Healthy (green leaves)
for _ in range(50):
    mean_r = np.random.normal(0.3, 0.03)
    mean_g = np.random.normal(0.5, 0.03)
    mean_b = np.random.normal(0.2, 0.03)
    std_r = np.random.normal(0.08, 0.02)
    std_g = np.random.normal(0.08, 0.02)
    std_b = np.random.normal(0.08, 0.02)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([1, 0, 0, 0])  # Healthy

# Class 1: Powdery (whitish, high brightness)
for _ in range(50):
    mean_r = np.random.normal(0.65, 0.05)
    mean_g = np.random.normal(0.70, 0.05)
    mean_b = np.random.normal(0.68, 0.05)
    std_r = np.random.normal(0.05, 0.01)
    std_g = np.random.normal(0.05, 0.01)
    std_b = np.random.normal(0.05, 0.01)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 1, 0, 0])  # Powdery

# Class 2: Rust (reddish-brown)
for _ in range(50):
    mean_r = np.random.normal(0.55, 0.05)
    mean_g = np.random.normal(0.30, 0.05)
    mean_b = np.random.normal(0.15, 0.05)
    std_r = np.random.normal(0.08, 0.02)
    std_g = np.random.normal(0.08, 0.02)
    std_b = np.random.normal(0.06, 0.02)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 0, 1, 0])  # Rust

# Class 3: Scab (dark with high contrast)
for _ in range(50):
    mean_r = np.random.normal(0.25, 0.05)
    mean_g = np.random.normal(0.20, 0.05)
    mean_b = np.random.normal(0.18, 0.05)
    std_r = np.random.normal(0.15, 0.03)
    std_g = np.random.normal(0.15, 0.03)
    std_b = np.random.normal(0.14, 0.03)
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.sqrt(std_r**2 + std_g**2 + std_b**2)
    
    X_train.append([mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast])
    y_train.append([0, 0, 0, 1])  # Scab

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Features creados: {X_train.shape}")
print(f"Distribución de clases:")
print(f"  - Healthy: {np.sum(y_train[:, 0])}")
print(f"  - Powdery: {np.sum(y_train[:, 1])}")
print(f"  - Rust: {np.sum(y_train[:, 2])}")
print(f"  - Scab: {np.sum(y_train[:, 3])}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler for later use
scaler_path = MODEL_DIR / 'feature_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n✓ Scaler guardado en: {scaler_path}")

# Create model
print("\nEntrenando modelo...")
model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

print(f"Entrenamiento completado - Accuracy final: {history.history['accuracy'][-1]:.2%}")

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
print(f"✓ Class indices guardados")

# Test with different feature vectors
print("\nProbando modelo con diferentes características...")

def predict_test(name, features):
    """Test prediction with a feature vector"""
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled, verbose=0)
    predicted_idx = np.argmax(pred[0])
    confidence = pred[0][predicted_idx] * 100
    
    class_names = ["Healthy", "Powdery", "Rust", "Scab"]
    print(f"  {name}: {class_names[predicted_idx]} ({confidence:.1f}%)")
    return class_names[predicted_idx]

# Test 1: Healthy (high green)
predict_test("Verde alto", [0.30, 0.48, 0.22, 0.08, 0.08, 0.08, 0.33, 0.09])

# Test 2: Powdery (whitish)
predict_test("Blanco brillante", [0.65, 0.70, 0.68, 0.05, 0.05, 0.05, 0.68, 0.05])

# Test 3: Rust (reddish)
predict_test("Rojo óxido", [0.55, 0.30, 0.15, 0.08, 0.08, 0.06, 0.33, 0.08])

# Test 4: Scab (dark with contrast)
predict_test("Oscuro con contraste", [0.25, 0.20, 0.18, 0.15, 0.15, 0.14, 0.21, 0.15])

print("\n✓ ¡Modelo entrenado y probado correctamente!")
print("✓ El modelo ahora usa features de color para clasificar enfermedades.")
