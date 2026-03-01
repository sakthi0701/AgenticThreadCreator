import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = os.getenv("gemini_api_key", "")
    gemini_api_key_2: str = os.getenv("gemini_api_key_2", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")
    
    # Rate Limiting
    daily_post_limit: int = 5
    free_tier_limit: int = 3
    pro_daily_limit: int = 3  # Pro Mode generations per IP per day
    
    # Model Configuration
    gemini_max_tokens: int = 8000
    gemini_temperature: float = 0.75
    gemini_max_retries: int = 3
    gemini_retry_delay: float = 2.0
    
    kimi_max_tokens: int = 2000
    kimi_temperature: float = 0.7
    
    # Training Configuration
    training_data_limit: int = 30
    max_labeled_demos: int = 2
    max_bootstrapped_demos: int = 20
    
    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "app.log"
    
    # Thread Pool Configuration
    supabase_logger_workers: int = 5
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        env_prefix = "APP_"

settings = Settings()
