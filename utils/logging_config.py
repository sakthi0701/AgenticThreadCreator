"""
Logging configuration for the AgenticThreadCreator application.
Provides structured logging with file and console handlers.
"""
import logging
import sys
from pathlib import Path
from config import settings


def setup_logging():
    """
    Configure application-wide logging based on settings.
    
    Creates both console and optional file handlers with consistent formatting.
    """
    # Create logger
    logger = logging.getLogger("agenticthreadcreator")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if settings.log_to_file:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


# Initialize logger
logger = setup_logging()


def get_logger(name: str = None):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"agenticthreadcreator.{name}")
    return logging.getLogger("agenticthreadcreator")
