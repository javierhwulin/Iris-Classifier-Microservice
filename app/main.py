from fastapi import FastAPI
from app.api.classifier import router as clf_router

def create_app():
    app = FastAPI(title="Iris Classifier")
    app.include_router(clf_router, prefix="/v1")
    return app