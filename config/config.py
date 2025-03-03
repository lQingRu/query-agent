from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HUGGINGFACE_API_KEY: str
    OLLAMA_URL: str

    class Config:
        env_file = ".env"


settings = Settings()
