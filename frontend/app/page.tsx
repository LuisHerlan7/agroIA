

"use client";

import React, { useState, useEffect, useRef } from "react";

export default function Home() {
  const [mensaje, setMensaje] = useState<string>("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [progress, setProgress] = useState<number>(0);
  const [uploading, setUploading] = useState<boolean>(false);
  const progressTimer = useRef<number | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/test")
      .then(res => res.json())
      .then(data => setMensaje(data.mensaje));
  }, []);
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Selecciona una imagen primero");
      return;
    }

    // Use XMLHttpRequest so we can track upload progress (0-100%)
    const formData = new FormData();
    formData.append("file", image);

    setUploading(true);
    setProgress(0);
    setResult("");
    // start a simulated progress that goes to 95% over ~5 seconds
    const duration = 5000; // ms
    const startTime = Date.now();
    if (progressTimer.current) {
      window.clearInterval(progressTimer.current);
    }
    progressTimer.current = window.setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(95, Math.round((elapsed / duration) * 95));
      setProgress(pct);
    }, 100) as unknown as number;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:8000/predict");

    xhr.upload.onprogress = (e: ProgressEvent) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        setProgress(percent);
      }
    };

    xhr.onload = () => {
      // stop simulated timer and finish progress
      if (progressTimer.current) {
        window.clearInterval(progressTimer.current);
        progressTimer.current = null;
      }
      setProgress(100);
      let parsed: any = null;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch (e) {
        // If parsing fails, show parse error after a short delay
        setTimeout(() => {
          setUploading(false);
          setResult(`Error al parsear respuesta: ${e}`);
        }, 900);
        return;
      }

      // Wait a bit to simulate processing time and match mockup behavior
      setTimeout(() => {
        setUploading(false);
        if (parsed.error || parsed.detail) {
          const msg = parsed.error ?? parsed.detail;
          setResult({ error: msg });
        } else if (parsed.disease_name) {
          setResult(parsed);
        } else {
          setResult({ error: "No se obtuvo respuesta" });
        }
      }, 1200);
    };

    xhr.onerror = () => {
      if (progressTimer.current) {
        window.clearInterval(progressTimer.current);
        progressTimer.current = null;
      }
      setUploading(false);
      setResult("Error de conexión al subir la imagen");
    };

    xhr.send(formData);
  };

  return (
    <main className="flex flex-col items-center gap-6 p-6 max-w-xl mx-auto bg-white rounded-lg shadow-md">
      <div>
        <h1>Respuesta de FastAPI</h1>
        <p>{mensaje}</p>
      </div>
      <h1 className="text-3xl font-bold text-green-700">
        Diagnóstico de Enfermedades en Hojas 🌿
      </h1>

      <p className="text-gray-600 text-center">
        Sube una foto de una hoja de planta y el sistema usará IA para detectar
        posibles enfermedades.
      </p>

      <label
        htmlFor="fileUpload"
        className="cursor-pointer bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg shadow-md"
      >
        Seleccionar imagen
      </label>

      <input
        id="fileUpload"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="hidden"
      />

      {/* Durante la carga mostramos una vista separada con la imagen y la barra debajo */}
      {uploading ? (
        <div className="w-full flex flex-col items-center mt-6">
          <div className="w-80 h-56 rounded-xl overflow-hidden shadow-lg">
            {preview ? (
              <img src={preview} alt="Preview" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-gray-100 flex items-center justify-center">No hay imagen</div>
            )}
          </div>

          <div className="w-full max-w-md mt-6">
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div
                className="bg-green-600 h-4 rounded-full"
                style={{ width: `${progress}%`, transition: 'width 200ms linear' }}
              />
            </div>
            <div className="text-center mt-2 font-medium">Cargando: {progress}%</div>
          </div>
        </div>
      ) : (
        <>
          {/* Vista previa de la imagen */}
          {preview && (
            <img
              src={preview}
              alt="Vista previa"
              className="w-64 h-64 object-cover rounded-lg shadow-lg"
            />
          )}

          {/* Botón para subir */}
          <button
            onClick={handleUpload}
            className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg shadow-md"
          >
            Analizar Imagen
          </button>
        </>
      )}

      {/* Resultado de la IA - Formato mockup */}
      {result && (
        <div className="mt-6 w-full max-w-2xl">
          {result.error ? (
            <div className="p-4 border border-red-300 bg-red-50 rounded-lg text-center">
              <h2 className="font-semibold text-lg text-red-700">Error:</h2>
              <p className="text-red-600">{result.error}</p>
            </div>
          ) : (
            <div className="p-6 border border-gray-300 rounded-lg bg-gradient-to-br from-gray-50 to-gray-100 shadow-md">
              {/* Estado y nombre de la enfermedad */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-700">Estado:</span>
                  <span className={`font-bold text-lg ${
                    result.disease_key === 'Healthy' 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {result.disease_name}
                  </span>
                </div>
                
                {/* Barra de confianza visual */}
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium text-gray-600 w-20">Barra de confianza:</span>
                  <div className="flex-1 bg-gray-300 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        result.confidence >= 80
                          ? 'bg-green-600'
                          : result.confidence >= 50
                          ? 'bg-yellow-500'
                          : 'bg-red-600'
                      }`}
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-gray-700 w-16 text-right">
                    {result.confidence}%
                  </span>
                </div>
              </div>

              {/* Descripción */}
              {result.description && (
                <div className="mb-6 p-4 bg-white rounded border border-gray-200">
                  <p className="text-gray-700 text-sm leading-relaxed">{result.description}</p>
                </div>
              )}

              {/* Sección de tratamiento */}
              {result.treatment && (
                <div className="mt-6 pt-6 border-t border-gray-300">
                  <h3 className="font-bold text-gray-800 mb-3">🌿 Sección de Tratamiento</h3>
                  <div className="bg-white p-4 rounded border-l-4 border-green-500">
                    <p className="font-semibold text-gray-800 mb-2">Tratamiento Recomendado:</p>
                    <p className="text-gray-700 text-sm leading-relaxed">{result.treatment}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </main>
  );
}
