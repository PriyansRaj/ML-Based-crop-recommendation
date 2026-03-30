from typing import Any
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from src.app import Main


app = FastAPI(title="Crop Recommendation API")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CropInput(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen value")
    P: float = Field(..., ge=0, description="Phosphorus value")
    K: float = Field(..., ge=0, description="Potassium value")
    temperature: float = Field(..., description="Temperature value")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="pH value")
    rainfall: float = Field(..., ge=0, description="Rainfall value")


class PredictionResponse(BaseModel):
    individual_predictions: dict[str, Any]
    final_prediction: list[str] | str


main_obj = Main()


@app.get("/")
def home():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "Crop Recommendation API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CropInput) -> PredictionResponse:
    try:
        payload = data.model_dump()
        result = main_obj.run(new_data=payload, decode=True)

        if isinstance(result, dict) and "individual_predictions" in result and "final_prediction" in result:
            serialized_predictions: dict[str, Any] = {}

            for model_name, pred in result["individual_predictions"].items():
                if isinstance(pred, np.ndarray):
                    serialized_predictions[model_name] = pred.tolist()
                else:
                    serialized_predictions[model_name] = pred

            final_prediction = result["final_prediction"]
            if isinstance(final_prediction, np.ndarray):
                final_prediction = final_prediction.tolist()

            return PredictionResponse(
                individual_predictions=serialized_predictions,
                final_prediction=final_prediction,
            )

        raise HTTPException(status_code=500, detail="Unexpected prediction response format")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/final")
def predict_final(data: CropInput) -> dict[str, Any]:
    try:
        payload = data.model_dump()
        result = main_obj.run(new_data=payload, decode=True)

        final_prediction = result.get("final_prediction")
        if isinstance(final_prediction, np.ndarray):
            final_prediction = final_prediction.tolist()

        return {"final_prediction": final_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/raw")
def predict_raw(data: CropInput) -> dict[str, Any]:
    try:
        payload = data.model_dump()
        result = main_obj.run(new_data=payload, decode=False)

        serialized_predictions: dict[str, Any] = {}
        for model_name, pred in result["individual_predictions"].items():
            if isinstance(pred, np.ndarray):
                serialized_predictions[model_name] = pred.tolist()
            else:
                serialized_predictions[model_name] = pred

        final_prediction = result["final_prediction"]
        if isinstance(final_prediction, np.ndarray):
            final_prediction = final_prediction.tolist()

        return {
            "individual_predictions": serialized_predictions,
            "final_prediction": final_prediction,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
