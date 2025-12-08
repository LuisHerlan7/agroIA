#!/usr/bin/env python
"""Test different disease types by creating synthetic test images"""
import requests
import numpy as np
from PIL import Image
from pathlib import Path

def create_test_image(name, rgb_values):
    """Create a test image with specific RGB values"""
    img_array = np.ones((224, 224, 3), dtype=np.uint8)
    for i, val in enumerate(rgb_values):
        img_array[:, :, i] = val
    
    img = Image.fromarray(img_array, mode='RGB')
    return img

print("Probando modelo con diferentes tipos de hoja...\n")

# Test 1: Healthy (green)
print("Test 1: HEALTHY (Hoja verde)")
img = create_test_image("healthy", [76, 128, 51])
img.save("test_healthy.jpg")
with open("test_healthy.jpg", 'rb') as f:
    res = requests.post('http://localhost:8000/predict', files={'file': f})
    data = res.json()
    print(f"  → {data['disease_name']} ({data['confidence']}%)")
    print()

# Test 2: Powdery (whitish)
print("Test 2: POWDERY (Hoja blanca/polvorienta)")
img = create_test_image("powdery", [179, 191, 184])
img.save("test_powdery.jpg")
with open("test_powdery.jpg", 'rb') as f:
    res = requests.post('http://localhost:8000/predict', files={'file': f})
    data = res.json()
    print(f"  → {data['disease_name']} ({data['confidence']}%)")
    print()

# Test 3: Rust (reddish)
print("Test 3: RUST (Hoja roja óxido)")
img = create_test_image("rust", [153, 89, 51])
img.save("test_rust.jpg")
with open("test_rust.jpg", 'rb') as f:
    res = requests.post('http://localhost:8000/predict', files={'file': f})
    data = res.json()
    print(f"  → {data['disease_name']} ({data['confidence']}%)")
    print()

# Test 4: Scab (dark)
print("Test 4: SCAB (Hoja oscura con manchas)")
img = create_test_image("scab", [76, 64, 58])
img.save("test_scab.jpg")
with open("test_scab.jpg", 'rb') as f:
    res = requests.post('http://localhost:8000/predict', files={'file': f})
    data = res.json()
    print(f"  → {data['disease_name']} ({data['confidence']}%)")
    print()

print("✓ Todas las pruebas completadas")
print("\nLimpiando archivos de prueba...")
for f in ["test_healthy.jpg", "test_powdery.jpg", "test_rust.jpg", "test_scab.jpg"]:
    Path(f).unlink(missing_ok=True)
print("✓ Hecho!")
