"""
FastAPI server for FinSights Application
"""

import logging
import time
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from models import HealthResponse
from api.routes import router
from services.observability_service import observability_service

# IMPORTANT: import the llm_service singleton object directly
from services.llm.llm_service import llm_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS.split(",") if config.CORS_ORIGINS != "*" else ["*"],
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# Include API routes
app.include_router(router)

@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    started = time.perf_counter()
    ctx_tokens = observability_service.set_request_context(request.url.path, request.method)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        observability_service.record_request(
            status_code=status_code,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        observability_service.reset_request_context(ctx_tokens)


@app.get("/")
def root():
    """Root endpoint with service info"""
    response = {
        "message": "FinSights Service is running",
        "version": config.APP_VERSION,
        "status": "healthy",
        "docs": "/docs",
        "health": "/health",
        "config": {
            "llm_provider": llm_service.get_provider_name(),
            "llm_model": llm_service.model,
            "api_token_configured": bool(config.API_TOKEN or config.OPENAI_API_KEY),
        },
    }
    return response


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Detailed health check"""
    response_data = {
        "status": "healthy",
        "service": config.APP_TITLE,
        "version": config.APP_VERSION,
    }

    llm_health = llm_service.health_check()

    response_data["llm_provider"] = llm_health.get("provider", llm_service.get_provider_name())

    # If OpenAI isn't configured or health check fails, mark unhealthy
    if llm_health.get("status") in ("not_configured", "unhealthy"):
        response_data["status"] = "unhealthy"

    return HealthResponse(**response_data)


@app.on_event("startup")
async def startup_event():
    """Log configuration on startup"""
    logger.info("=" * 60)
    logger.info(f"Starting {config.APP_TITLE} v{config.APP_VERSION}")
    logger.info("=" * 60)
    logger.info(f"LLM Provider: {llm_service.get_provider_name()}")
    logger.info(f"API Token Configured: {bool(config.API_TOKEN or config.OPENAI_API_KEY)}")
    logger.info(f"Model: {llm_service.model}")
    logger.info(f"Port: {config.SERVICE_PORT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        timeout_keep_alive=300,
    )
