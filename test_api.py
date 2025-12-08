#!/usr/bin/env python
"""Create a test image and send to backend"""
import requests
from PIL import Image
import numpy as np
from pathlib import Path
import io

print("Creando imagen de prueba...")

# Create a simple test image (224x224 RGB)
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array, mode='RGB')

# Save temporarily
test_img_path = Path(__file__).parent / 'test_image.jpg'
img.save(test_img_path)
print(f"✓ Imagen de prueba guardada: {test_img_path}")

# Send to backend
print("\nEnviando a backend...")
try:
    with open(test_img_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict',
            files={'file': f}
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")
    
    if response.status_code == 200:
        print(f"\n✓ ¡Éxito! Resultado: {data['resultado']} (Confianza: {data['confianza']}%)")
    else:
        print(f"\n✗ Error: {data}")
except requests.exceptions.ConnectionError:
    print("✗ Error: No se pudo conectar al backend. ¿Está ejecutando?")
except Exception as e:
    print(f"✗ Error: {e}")
finally:
    # Clean up
    test_img_path.unlink(missing_ok=True)
