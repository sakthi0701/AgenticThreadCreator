import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("gemini_api_key", "")
    gemini_api_key_2: str = os.getenv("gemini_api_key_2", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
