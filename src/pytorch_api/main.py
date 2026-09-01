
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import time

from .model import model, DEVICE
from .preprocessing import preprocess_image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@app.get("/")
def home():
    return {
        "message": "Fashion MNIST API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:

        start_time = time.perf_counter()

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        image_tensor = preprocess_image(image)

        X = image_tensor.reshape(
            image_tensor.shape[0],
            -1
        )

        X = X.to(DEVICE)

        with torch.no_grad():

            y_pred_prob = model(X)

            probabilities = torch.softmax(
                y_pred_prob,
                dim=1
            )

            y_pred = torch.argmax(
                probabilities,
                dim=1
            )

        prediction = int(y_pred.item())

        confidence = float(
            probabilities[0][prediction].item()
        )

        scores = {
            CLASS_NAMES[i]: float(probabilities[0][i].item())
            for i in range(len(CLASS_NAMES))
        }

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        return {
            "predictedClass": CLASS_NAMES[prediction],
            "confidence": confidence,
            "scores": scores,
            "latencyMs": latency_ms
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

