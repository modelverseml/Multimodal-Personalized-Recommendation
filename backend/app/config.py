from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_MINUTES: int = 60
    CORS_ORIGINS: str = "http://localhost:5174,http://127.0.0.1:5174"

    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    DATASET_CATEGORY: str = "Video_Games"
    ARTIFACT_DIR: str = "artifacts"
    IMAGE_DIR: str = "data/images"
    RAW_DIR: str = "data/raw"
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT: str = "multimodal-recsys"

    # Online serving knobs
    RETRIEVAL_CANDIDATES: int = 200
    MAX_HISTORY: int = 50
    DRIFT_PSI_THRESHOLD: float = 0.2

    DEMO_EMAIL: str | None = None
    DEMO_PASSWORD: str | None = None

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        extra="ignore",
    )

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else BACKEND_DIR / p

    @property
    def artifact_dir(self) -> Path:
        return self.path(self.ARTIFACT_DIR)

    @property
    def image_dir(self) -> Path:
        return self.path(self.IMAGE_DIR)

    @property
    def raw_dir(self) -> Path:
        return self.path(self.RAW_DIR)

    @property
    def mlflow_uri(self) -> str:
        # Resolve relative sqlite paths against the backend dir so scripts
        # and the API agree regardless of the working directory.
        prefix = "sqlite:///"
        uri = self.MLFLOW_TRACKING_URI
        if uri.startswith(prefix) and not uri[len(prefix):].startswith("/"):
            return prefix + str(BACKEND_DIR / uri[len(prefix):])
        return uri


settings = Settings()
