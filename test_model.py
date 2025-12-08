#!/usr/bin/env python
"""Test script to debug model loading"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

print("Testing model and class indices loading...")
print(f"Working directory: {Path.cwd()}")

try:
    from app.utils import load_model, load_class_indices
    print("\n✓ Imported utils successfully")
    
    print("\nLoading model...")
    model = load_model()
    print("✓ Model loaded successfully")
    
    print("\nLoading class indices...")
    indices = load_class_indices()
    print(f"✓ Class indices loaded: {len(indices)} clases")
    for k, v in list(indices.items())[:5]:
        print(f"  - {k}: {v}")
    
    print("\n✓ All checks passed!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
