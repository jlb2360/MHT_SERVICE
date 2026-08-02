import os
import uvicorn

def main() -> None:
    """
    Main entry point for the Multiple Hypothesis Tracking (MHT) microservice.
    """
    # Configuration via environment variables with sensible defaults
    host = os.getenv("MHT_API_HOST", "0.0.0.0")
    port = int(os.getenv("MHT_API_PORT", "8000"))
    workers = int(os.getenv("MHT_API_WORKERS", "1"))
    
    # Enable hot-reloading for local development if specified
    reload = os.getenv("MHT_API_RELOAD", "False").lower() in ("true", "1", "t", "yes")

    print(f"Starting MHT service on {host}:{port} with {workers} worker(s)...")

    # Run the FastAPI app
    # The string format "module.path:app_instance" is required when using reload/workers
    uvicorn.run(
        "mht_service.services.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()