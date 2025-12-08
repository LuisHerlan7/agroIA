

"use client";

import React, { useState, useEffect } from "react";

export default function Home() {
    const [mensaje, setMensaje] = useState<string>("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");

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

    try {
      const formData = new FormData();
      formData.append("file", image);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      
      if (data.error) {
        setResult(`Error: ${data.error}`);
      } else if (data.resultado) {
        setResult(`${data.resultado} (Confianza: ${data.confianza}%)`);
      } else {
        setResult("No se obtuvo respuesta");
      }
    } catch (error) {
      setResult(`Error de conexión: ${error}`);
    }
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

      {/* Resultado de la IA */}
      {result && (
        <div className="mt-4 p-4 border border-gray-300 rounded-lg w-full text-center">
          <h2 className="font-semibold text-lg">Resultado:</h2>
          <p className="text-gray-800">{result}</p>
        </div>
      )}
    </main>
  );
}
