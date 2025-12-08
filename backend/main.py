from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import tempfile
import imghdr
import os
import logging
import traceback
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from app.utils import predict_disease

app = FastAPI()

# si hay deploy añadir el link del front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/test")
def hola():
    return {"mensaje": "El servidor está funcionando correctamente"}

@app.get("/")
def root():
    return {"message": "Root, Hola desde FastAPI"}

@app.get("/api/health")
def health_check():
    """Verifica que el modelo y los índices de clase estén accesibles"""
    from app.utils import load_model, load_class_indices
    
    status = {"status": "ok", "checks": {}}
    
    try:
        model = load_model()
        status["checks"]["model"] = "loaded"
    except Exception as e:
        status["status"] = "error"
        status["checks"]["model"] = str(e)
    
    try:
        indices = load_class_indices()
        status["checks"]["class_indices"] = f"loaded ({len(indices)} clases)"
    except Exception as e:
        status["status"] = "error"
        status["checks"]["class_indices"] = str(e)
    
    return status

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío o no recibido correctamente")

    try:
        result = predict_disease(image_bytes)
        return {
            "disease_key": result["disease_key"],
            "disease_name": result["disease_name"],
            "description": result["description"],
            "treatment": result["treatment"],
            "confidence": round(result["confidence"] * 100, 2)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        try:
            tb_text = traceback.format_exc()
            log_line = f"\n---- {datetime.utcnow().isoformat()} UTC ----\n{tb_text}\n"
            with open('backend_error.log', 'a', encoding='utf-8') as lf:
                lf.write(log_line)
        except Exception:
            pass
        logging.exception("Error en predict:")
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")
