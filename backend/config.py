"""
Configuration settings for Doc-Sum Application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")



# LLM Configuration (tuned for section summaries)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))


# Application Settings
APP_TITLE = "FinSights"
APP_DESCRIPTION = "AI-powered financial document summarization with GPT models."
APP_VERSION = "1.0.0"

# Service Configuration
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File Upload Settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(500 * 1024 * 1024)))  # 500MB
MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE", str(50 * 1024 * 1024)))  # 50MB

# File Processing Limits
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))  # Maximum pages to process from PDF
WARN_PDF_PAGES = 50  # Warn user if PDF has more than this many pages

# CORS Settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
