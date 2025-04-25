from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_path: str = "models/iris_net.pt"

    model_config = SettingsConfigDict(
        env_file = ".env",
        extra="ignore"
    )
    
settings = Settings()