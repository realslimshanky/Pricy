import os
import pickle
import pandas as pd
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from models import PricePredictionInput, PricePredictionOutput

load_dotenv()

MODEL_FILENAME = os.getenv("MODEL_FILENAME")
if not MODEL_FILENAME:
    raise ValueError("Set MODEL_FILENAME to .env")


with open(MODEL_FILENAME, "rb") as f_in:
    preprocessor, model = pickle.load(f_in)


app = FastAPI()


@app.post("/predict_price", response_model=PricePredictionOutput)
async def predict_price(
    data: PricePredictionInput,
):
    """
    Predict nightly rental price based on listing features and amenities.
    """

    df = pd.DataFrame([data.model_dump()])
    X = preprocessor.transform(df)
    price_pred = model.predict(X)[0]

    return {
        "predicted_price": float(price_pred)
    }
