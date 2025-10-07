from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    OPENAI_API_KEY: str
    VECTOR_STORE_URI: str
    VECTOR_STORE_COLLECTION_NAME: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
