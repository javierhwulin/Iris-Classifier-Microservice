from fastapi import APIRouter
from app.api.schemas import IrisInput, IrisOutput
from app.core.model import get_model
import joblib
import numpy as np
import torch

router = APIRouter()

@router.post("/predict", response_model=IrisOutput)
def predict(input: IrisInput):
    # load model + scaler
    model = get_model()
    scaler = joblib.load("models/scaler.pkl")

    x = np.array([[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]], dtype="float32")

    x_scaled = scaler.transform(x)

    logits = model(torch.from_numpy(x_scaled))
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    idx = int(probs.argmax())

    names = ["setosa", "versicolor", "virginica"]
    return IrisOutput(class_name=names[idx], confidence=float(probs[idx]))
