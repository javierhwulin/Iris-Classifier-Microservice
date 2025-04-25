import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.core.model import IrisNet
import joblib

def main():
    # load and preprocess
    iris = load_iris()
    X, y = iris.data.astype("float32"), iris.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler, "models/scaler.pkl")

    # make data loader
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)

    # build model
    model = IrisNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    # save trained weights
    torch.save(model.state_dict(), "models/iris_net.pt")
    print("Training complete - model saved to models/iris_net.pt")

if __name__ == "__main__":
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    main()
