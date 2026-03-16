from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from App.recognition_log import get_logs, add_log
from App.vector_store import add_face, search_face

app = FastAPI()

# ensure images folder exists
os.makedirs("images", exist_ok=True)


@app.get("/")
def home():
    return {"message": "Face Recognition API running"}


# Cloud-safe register endpoint
@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # recognition disabled in cloud
    return {
        "message": "Face registration is disabled on cloud deployment"
    }


# Cloud-safe recognition endpoint
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):

    return {
        "recognized": "Recognition disabled on cloud deployment"
    }


# logs endpoint
@app.get("/logs")
def recognition_logs():
    return get_logs()


# dashboard recognition endpoint
@app.post("/recognize_frame")
async def recognize_frame(file: UploadFile = File(...)):

    name = "Recognition disabled"

    add_log(name)

    return {"name": name}


# dashboard page
@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")