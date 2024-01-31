import cv2
import const
import os
from pathlib import Path
from fastapi import FastAPI, File, Request
from fastapi.staticfiles import StaticFiles

app = FastAPI()

const.FOLDER_CURRENT = os.path.dirname(os.path.abspath(__file__))
const.FOLDER_CASCADES = os.path.join(const.FOLDER_CURRENT, "model")
const.FOLDER_IMAGES = os.path.join(const.FOLDER_CURRENT, "images")
const.PATH_FACE_DETECTOR = os.path.join(
    const.FOLDER_CURRENT, const.FOLDER_CASCADES + "/haarcascade_frontalface_default.xml"
)
const.SAMPLE_IMG_PATH = os.path.join(
    const.FOLDER_CURRENT, const.FOLDER_IMAGES, "robert.jpg"
)

@app.get("/")
async def root():
    img = cv2.imread(const.SAMPLE_IMG_PATH)
    print(const.SAMPLE_IMG_PATH)

    if img is None:
        return {"message": "image not found"}

    face_classifier = cv2.CascadeClassifier(const.PATH_FACE_DETECTOR)

    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    box, detections, weight = face_classifier.detectMultiScale3(
        gray, minNeighbors=8, outputRejectLevels=True
    )

    print(detections)
    print(box)

    return {"message": "success", "detections": "", "box": "", "score": weight[0]}

