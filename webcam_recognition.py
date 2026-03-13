import cv2
import face_recognition

from App.vector_store import search_face

video_capture = cv2.VideoCapture(0)

print("Starting webcam... Press Q to quit")

process_this_frame = True

while True:

    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # resize frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # scale back coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        name = search_face(face_encoding)

        if not name:
            name = "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()