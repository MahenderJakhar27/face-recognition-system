from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import tempfile

from App.recognition_log import get_logs, add_log
from App.vector_store import add_face, search_face
from App.face_service import get_face_embedding

app = FastAPI()

# ensure images folder exists
os.makedirs("images", exist_ok=True)


@app.get("/")
def home():
    return {"message": "Face Recognition API running"}


# Register endpoint
@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = get_face_embedding(file_path)

    if embedding is None:
        return {"message": "No face detected in image, please try another photo"}

    add_face(embedding, name)

    return {"message": f"{name} registered successfully"}


# Recognition endpoint
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    embedding = get_face_embedding(tmp_path)
    os.remove(tmp_path)

    if embedding is None:
        return {"recognized": "No face detected"}

    result = search_face(embedding)
    name = result if result else "Unknown"

    return {"recognized": name}


# logs endpoint
@app.get("/logs")
def recognition_logs():
    return get_logs()


# dashboard recognition endpoint
@app.post("/recognize_frame")
async def recognize_frame(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    embedding = get_face_embedding(tmp_path)
    os.remove(tmp_path)

    if embedding is None:
        name = "No face detected"
    else:
        result = search_face(embedding)
        name = result if result else "Unknown"

    add_log(name)

    return {"name": name}


# dashboard page
@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")