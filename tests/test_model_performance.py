import torch
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from app.core.model import IrisNet
from pathlib import Path

@pytest.fixture(scope="module")
def trained_model_path():
    path = Path("models/iris_net.pt")
    assert path.exists(), f"Model file not found at {path}"
    return path

def test_accuracy_above_threshold(trained_model_path):
    # Load model
    model = IrisNet()
    state = torch.load(trained_model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Prepare test data
    iris = load_iris()
    X, y = iris.data.astype("float32"), iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)

    # Inference
    with torch.no_grad():
        inputs = torch.from_numpy(X_test)
        logits = model(inputs)
        preds = logits.argmax(dim=1).numpy()

    acc = (preds == y_test).mean()
    # Assert at least 95% accuracy
    assert acc >= 0.95, f"Model accuracy too low: {acc:.2%}"
