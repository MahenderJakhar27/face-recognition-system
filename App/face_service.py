from deepface import DeepFace
import numpy as np


def get_face_embedding(image_path):
    """
    Detects face in an image and returns the face embedding vector
    """

    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )

        if not embedding_obj:
            return None

        return np.array(embedding_obj[0]["embedding"])

    except Exception as e:
        print(f"Embedding error: {e}")
        return None