como iniciar el proyecto
primero, necesario tener python
segundo, necesario tener npm
---------------------------
BACKEND
------------------------

cd backend

(crear e iniciar un entorno virtual para evitar errores)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate   # Windows xd

pip install fastapi uvicorn
pip install fastapi[all]

uvicorn main:app --reload --port 8000

El link deberia ser http://127.0.0.1:8000/
tambien funciona http://localhost:8000/

------------------------------
FRONTEND
-------------------------------

cd frontend

npm install
npm run dev

el link deberia ser http://localhost:3000