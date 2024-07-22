from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

@app.post("/detect")
async def detect_faces(imageFile: UploadFile = File(...)):
    # Save the uploaded image temporarily
    image_data = await imageFile.read()

    # Forward the image data to the face detection Python app
    try:
        response = requests.post('https://fdet-new.vercel.app/face_detection_app', files={'imageFile': image_data})
        response.raise_for_status()
        detection_results = response.json()
        return detection_results
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
