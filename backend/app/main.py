
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pathlib import Path
from backend.app.core.config import settings
from backend.app.api.routers import chat, upload, health

# Configure logging
logger.add(
    settings.LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    rotation="50 MB",
    retention="14 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)

app = FastAPI(title="Hybrid Medical RAG Chatbot", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)

# Mount static files and serve frontend
app.mount("/static", StaticFiles(directory=str(settings.FRONTEND_DIR / "static")), name="static")


@app.get("/")
async def read_index():
    """Serve the frontend index.html."""
    return FileResponse(str(settings.FRONTEND_DIR / "index.html"))


# Backward compatibility endpoints
@app.post("/chat")
async def old_chat_endpoint(request: Request):
    """Redirect to new chat endpoint with backward compatibility."""
    from fastapi.responses import JSONResponse
    from backend.app.services.rag_service import rag_pipeline
    from backend.app.services.llm_router import generate_general_response
    from loguru import logger

    pipeline = rag_pipeline

    try:
        body = await request.json()
        question = body.get("question")
        logger.info(f"Received question (old endpoint): {question}")
        if pipeline:
            response = pipeline.run(question)
        else:
            response = generate_general_response(question)
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Chat error (old endpoint): {e}")
        try:
            body = await request.json()
            question = body.get("question")
            response = generate_general_response(question)
            return JSONResponse(content={"response": response})
        except:
            error_msg = "I'm sorry, I encountered an internal error. Please try again later."
            if "quota" in str(e).lower() or "429" in str(e):
                error_msg = "The AI service is currently at its limit (Quota Exceeded). Please wait a moment."
            return JSONResponse(content={"response": error_msg})


@app.post("/upload")
async def old_upload_endpoint(request: Request):
    """Redirect to new upload endpoint with backward compatibility."""
    from fastapi import UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    from loguru import logger
    import os
    import shutil
    from backend.app.core.config import settings
    from backend.app.services.rag_service import rag_pipeline

    pipeline = rag_pipeline

    try:
        form = await request.form()
        file: UploadFile = form.get("file")
        if not file:
            return JSONResponse(content={"status": "error", "message": "No file provided"})

        if not pipeline:
            return JSONResponse(content={"status": "error", "message": "RAG pipeline not available, cannot ingest PDFs"})

        file_path = os.path.join(str(settings.UPLOADS_DIR), file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded (old endpoint): {file.filename}. Starting ingestion...")
        report = pipeline.ingest_pdfs(str(settings.UPLOADS_DIR))

        if report["chunks_added"] == 0:
            return JSONResponse(content={
                "answer": answer,
                "mode": "rag" if used_rag else "general_llm_fallback",
                "source": "gemini",
            })

        return JSONResponse(content={
            "status": "success",
            "message": f"File {file.filename} uploaded and ingested ({report['chunks_added']} chunks).",
            "details": report
        })
    except Exception as e:
        logger.error(f"Upload error (old endpoint): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/kb-status")
async def old_kb_status_endpoint():
    """Backward compatibility for kb status endpoint."""
    from backend.app.services.rag_service import rag_pipeline
    from loguru import logger

    pipeline = rag_pipeline

    if not pipeline:
        return {"status": "no pipeline"}
    return {"total_vectors": pipeline.chroma.count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=False)

