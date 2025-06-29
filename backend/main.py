from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
features = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

@app.get("/")
def read_root():
    return {"message": "Welcome to Predict API"}


class CancerFeatures(BaseModel):
    Clump_Thickness: int
    Cell_Size_Uniformity: int
    Cell_Shape_Uniformity: int
    Marginal_Adhesion: int
    Single_Epi_Cell_Size: int
    Bare_Nuclei: int
    Bland_Chromatin: int
    Normal_Nucleoli: int
    Mitoses: int

@app.post("/predict")
def predict(features_input: CancerFeatures):
    
    data = pd.DataFrame([[
        features_input.Clump_Thickness,
        features_input.Cell_Size_Uniformity,
        features_input.Cell_Shape_Uniformity,
        features_input.Marginal_Adhesion,
        features_input.Single_Epi_Cell_Size,
        features_input.Bare_Nuclei,
        features_input.Bland_Chromatin,
        features_input.Normal_Nucleoli,
        features_input.Mitoses
    ]], columns=features)

    prediction = model.predict(data)[0]
    return {
        "prediction": int(prediction),
        "diagnosis": "Cancer" if prediction == 1 else "No Cancer"
    }
