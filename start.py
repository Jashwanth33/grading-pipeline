import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WEB_CONCURRENCY", 2))

    print(f"Starting server on {host}:{port} with {workers} workers")
    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
