from fastapi import FastAPI
from app.api.classifier import router as clf_router

def create_app():
    app = FastAPI(title="Iris Classifier")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(clf_router, prefix="/v1")    
    return app
 