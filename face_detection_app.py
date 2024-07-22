from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

app = FastAPI()

@app.post("/face_detection_app")
async def detect_faces(imageFile: UploadFile = File(...)):
    # Read image file
    image_bytes = await imageFile.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Perform face detection (example using OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    # Example response
    detection_results = {
        "faces_detected": len(faces),
        "image_width": img.shape[1],
        "image_height": img.shape[0]
    }
    return detection_results
