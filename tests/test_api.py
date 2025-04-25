from fastapi.testclient import TestClient
from app.main import create_app

client = TestClient(create_app())

def test_predict_setosa():
    sample = { "sepal_length": 5.1, 
              "sepal_width": 3.5, 
              "petal_length": 1.4, 
              "petal_width": 0.2 }
    
    r = client.post("/v1/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] == "setosa"
    assert 0.0 <= body["confidence"] <= 1.0