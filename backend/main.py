from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and related assets
model = joblib.load("model/model.joblib")
encoders = joblib.load("model/encoders.joblib")
features = joblib.load("model/feature_names.joblib")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Predict API"}

# Define the input schema
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

# Prediction endpoint
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
