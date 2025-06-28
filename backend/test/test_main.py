from fastapi.testclient import TestClient
from backend.main import app  

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200 
    assert response.json() == {"message": "Welcome to Predict API"}

def test_predict():
    payload = {
        "Clump_Thickness": 5,
        "Cell_Size_Uniformity": 1,
        "Cell_Shape_Uniformity": 1,
        "Marginal_Adhesion": 1,
        "Single_Epi_Cell_Size": 2,
        "Bare_Nuclei": 1,
        "Bland_Chromatin": 3,
        "Normal_Nucleoli": 1,
        "Mitoses": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "diagnosis" in data
    assert data["diagnosis"] in ["Cancer", "No Cancer"]
