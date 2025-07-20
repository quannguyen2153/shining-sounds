from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    allowed_origins: list[str] = ["*"]

    separation_model: str = "htdemucs"
    stem_ttl_seconds: int = 60
    cleanup_interval: int = 60

    transcription_model: str = "medium"

settings = Settings()