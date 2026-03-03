"""
Configuration settings for FinSights Application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional fallback model when MODEL_NAME is not set
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Unified Provider Configuration (primary)
API_ENDPOINT = os.getenv("API_ENDPOINT", "")
API_TOKEN = os.getenv("API_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
PROVIDER_NAME = os.getenv("PROVIDER_NAME", "")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true")
LOCAL_URL_ENDPOINT = os.getenv("LOCAL_URL_ENDPOINT", "not-needed")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "same")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "")
EMBEDDING_API = os.getenv("EMBEDDING_API", "")
EMBEDDING_API_ENDPOINT = os.getenv("EMBEDDING_API_ENDPOINT", "")
EMBEDDING_API_TOKEN = os.getenv("EMBEDDING_API_TOKEN", "")
EMBEDDING_PROVIDER_NAME = os.getenv("EMBEDDING_PROVIDER_NAME", "")

# Optional embedding overrides (advanced)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
INFERENCE_EMBEDDING_MODEL_NAME = os.getenv("INFERENCE_EMBEDDING_MODEL_NAME", "")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "")

# Legacy compatibility (optional)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
INFERENCE_API_ENDPOINT = os.getenv("INFERENCE_API_ENDPOINT", "")
INFERENCE_API_TOKEN = os.getenv("INFERENCE_API_TOKEN", "")
INFERENCE_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "")
OLLAMA_TOKEN = os.getenv("OLLAMA_TOKEN", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")



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
