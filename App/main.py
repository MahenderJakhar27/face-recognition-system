from fastapi import FastAPI, UploadFile, File
import shutil
from App.recognition_log import get_logs
from App.face_service import get_face_embedding
from App.vector_store import add_face, search_face
from fastapi import UploadFile, File
from App.recognition_log import add_log

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Face Recognition API running"}


@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = get_face_embedding(file_path)

    if embedding is None:
        return {"error": "No face detected"}

    add_face(embedding, name)

    return {"message": f"{name} registered successfully"}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = get_face_embedding(file_path)

    if embedding is None:
        return {"error": "No face detected"}

    name = search_face(embedding)

    if name:
        return {"recognized": name}

    return {"recognized": "Unknown"}

@app.get("/logs")
def recognition_logs():
    return get_logs()

@app.post("/recognize_frame")
async def recognize_frame(file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = get_face_embedding(file_path)

    if embedding is None:
        return {"name": "Unknown"}

    name = search_face(embedding)

    if not name:
        name = "Unknown"

    add_log(name)

    return {"name": name}