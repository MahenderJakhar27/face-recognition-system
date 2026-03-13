import face_recognition


def get_face_embedding(image_path):
    """
    Detects face in an image and returns the face embedding vector
    """

    # Load image
    image = face_recognition.load_image_file(image_path)

    # Detect faces
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return None

    # Generate embeddings
    encodings = face_recognition.face_encodings(image, face_locations)

    return encodings[0]