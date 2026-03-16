import faiss
import numpy as np
import os
import pickle

INDEX_FILE = "face_index.faiss"
LABEL_FILE = "face_labels.pkl"

# load existing index if present
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(128)

# load labels
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "rb") as f:
        labels = pickle.load(f)
else:
    labels = []


def save_index():
    faiss.write_index(index, INDEX_FILE)

    with open(LABEL_FILE, "wb") as f:
        pickle.dump(labels, f)


def add_face(embedding, name):

    vector = np.array([embedding]).astype("float32")

    index.add(vector)

    labels.append(name)

    save_index()


def search_face(embedding):

    if len(labels) == 0:
        return None

    vector = np.array([embedding]).astype("float32")

    # search top 5 matches
    D, I = index.search(vector, 5)

    distances = D[0]
    indexes = I[0]

    threshold = 0.35
    candidates = []

    for dist, idx in zip(distances, indexes):

        if idx < len(labels) and dist < threshold:
            candidates.append(labels[idx])

    if len(candidates) == 0:
        return None

    # majority vote
    from collections import Counter
    most_common = Counter(candidates).most_common(1)

    return most_common[0][0]