from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        enfermedad, confianza = predict_disease(image_bytes)
        
        return {
            "resultado": enfermedad,
            "confianza": round(confianza * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}
