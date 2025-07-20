from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    allowed_origins: list[str] = ["*"]

    stem_ttl_seconds: int = 60
    cleanup_interval: int = 60

settings = Settings()