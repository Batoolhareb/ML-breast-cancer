import pytest
import sys
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

@pytest.fixture
def client(monkeypatch):
    # Set required environment variables
    monkeypatch.setenv("MODELNAME", "test_model")
    monkeypatch.setenv("MODELVERSION", "1")
    
    # Mock MLflow modules
    mlflow_mock = MagicMock()
    mlflow_pyfunc = MagicMock()
    mlflow_mock.pyfunc = mlflow_pyfunc
    sys.modules['mlflow'] = mlflow_mock
    sys.modules['mlflow.pyfunc'] = mlflow_pyfunc
    
    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # Cancer prediction
    mlflow_pyfunc.load_model.return_value = mock_model
    
    # Force reload main to apply mocks
    if 'main' in sys.modules:
        del sys.modules['main']
    from main import app
    return TestClient(app)

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Cancer Prediction API (via MLflow model)"}

def test_predict(client):
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
    assert data == {
        "prediction": 1,
        "diagnosis": "Cancer"
    }