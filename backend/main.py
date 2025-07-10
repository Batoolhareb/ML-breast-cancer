from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Add this line
import mlflow
import mlflow.sklearn
import os

import mlflow.pyfunc

app = FastAPI()
data = pd.read_csv("../dataset/dataset.csv")
label_class = LabelEncoder()
data["Class"] = label_class.fit_transform(data["Class"])


#model = joblib.load("../model-storage/forest.joblib")

# MLflow model URI


model_name = os.getenv("MODELNAME")
model_version = os.getenv("MODELVERSION")

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")


# Manually define expected features (or load from joblib if saved separately)
EXPECTED_FEATURES = [
    "Clump_Thickness",
    "Cell_Size_Uniformity",
    "Cell_Shape_Uniformity",
    "Marginal_Adhesion",
    "Single_Epi_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses"
]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cancer Prediction API (via MLflow model)"}


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
    ]], columns=EXPECTED_FEATURES)

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction),
        "diagnosis": "Cancer" if prediction == 1 else "No Cancer"
    }
