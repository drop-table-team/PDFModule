import json
import os
import tempfile

import aiohttp
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader

from app.config import get_settings
from app.models.schemas import DocumentMetadata
from app.pipeline.pdf_pipeline import PDFAnalysisPipeline
from app.util.llm_provider import LLMFactory, LLMProvider

router = APIRouter()


def get_llm_provider() -> LLMProvider:
    return LLMFactory.create_provider(
        "ollama", get_settings().OLLAMA_BASE_URL, get_settings().OLLAMA_MODEL
    )


@router.get("/health")
async def health_check(llm_provider: LLMProvider = Depends(get_llm_provider)):
    """Basic health check that also verifies LLM connection."""
    try:
        response = await llm_provider.generate("Test connection")
        return {
            "status": "healthy",
            "llm_status": "connected",
            "llm_response": response,
        }
    except Exception as e:
        return {"status": "healthy", "llm_status": "error", "llm_error": str(e)}


@router.post("/input", response_model=DocumentMetadata)
async def process_document(
    file: UploadFile = File(...),
    llm_provider: LLMProvider = Depends(get_llm_provider),
):
    """
    Process a PDF document and forward the results to the backend service.
    Returns the generated metadata and forwards both file and metadata to the configured backend.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Create a temporary file to store the uploaded document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

            # Load the PDF and split it into pages
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()

            # Process the document with the analysis pipeline
            pipeline = PDFAnalysisPipeline(llm_provider)
            document_metadata = await pipeline.process_document(pages)

            backend_url = f"{get_settings().BACKEND_BASE_URL}/modules/input"
            json_payload = {
                "title": document_metadata.title,
                "tags": document_metadata.tags,
                "short": document_metadata.short_summary,
                "transcription": document_metadata.summary,
            }

            # Send multipart request to backend
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field(
                    "json", json.dumps(json_payload), content_type="application/json"
                )
                await file.seek(0)
                form.add_field(
                    "file",
                    await file.read(),
                    filename=file.filename,
                    content_type=file.content_type,
                )

                async with session.post(backend_url, data=form) as response:
                    if response.status >= 400:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Backend service error: {await response.text()}",
                        )
                    pass

            return document_metadata

    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error communicating with backend service: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Ensure the temporary file is deleted
        try:
            os.unlink(tmp_file.name)
        except Exception as e:
            # Log or handle any issues with file deletion if necessary
            print(f"Warning: Failed to delete temporary file {tmp_file.name}: {str(e)}")
