"""
API Routes for Doc-Sum Application
Handles all HTTP endpoints
"""

from fastapi import APIRouter, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional
import os
import logging
import json

import config
from models import HealthResponse

from services import pdf_service, llm_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - OpenAI-only"""
    llm_health = llm_service.health_check()

    response = {
        "status": "healthy" if llm_health.get("status") == "healthy" else "unhealthy",
        "service": config.APP_TITLE,
        "version": config.APP_VERSION,
        "llm_provider": "OpenAI",
    }

    return response


@router.post("/v1/docsum")
async def summarize_document(
    background_tasks: BackgroundTasks,
    type: str = Form(...),
    messages: str = Form(""),
    max_tokens: int = Form(1024),
    language: str = Form("en"),
    summary_type: str = Form("auto"),
    stream: str = Form("false"),
    # section-based controls
    mode: str = Form("financial_initial"),  # financial_initial | financial_section | financial_overall | financial_sectionwise
    section: str = Form(""),                # required if mode=financial_section
    # doc_id for cached section requests (frontend should send this on clicks)
    doc_id: str = Form(""),
    files: Optional[UploadFile] = File(None),
):
    """
    Summarize text or document content (PDF/DOC/DOCX/TXT)

    Supports:
    - Cached path: doc_id present -> no need to resend file
    - File path: file present -> creates doc_id; if mode=financial_section returns that section immediately
    """
    try:
        stream_bool = stream.lower() == "true"

        # Enforce backend constraints:
        # streaming supported only for financial_overall (per llm_service)
        if stream_bool and mode != "financial_overall":
            stream_bool = False

        if mode == "financial_section" and not section.strip():
            raise HTTPException(status_code=400, detail="section is required when mode=financial_section")

        logger.info(
            f"Request received - type={type}, has_file={files is not None}, "
            f"doc_id={doc_id}, messages_len={len(messages)}, mode={mode}, section={section}, stream={stream_bool}"
        )

        # ========== Cached Doc ID Path ==========
        # User clicks on sections: frontend should send doc_id + mode=financial_section + section
        if doc_id.strip():
            try:
                summary = llm_service.summarize_by_doc_id(
                    doc_id=doc_id.strip(),
                    max_tokens=max_tokens,
                    stream=stream_bool,
                    mode=mode,
                    section=section.strip() if section else None,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            if stream_bool:
                return StreamingResponse(_format_stream(summary), media_type="text/event-stream")

            return {
                "doc_id": doc_id.strip(),
                "text": summary,
                "summary": summary,
                "word_count": len(summary.split()),
                "char_count": len(summary),
                "mode": mode,
                "section": section.strip() if section else "",
            }

        # ========== Text Input ==========
        if type == "text" and messages.strip():
            logger.info("Processing text input")
            summary = llm_service.summarize(
                text=messages,
                max_tokens=max_tokens,
                stream=stream_bool,
                mode=mode,
                section=section.strip() if section else None,
            )

            if stream_bool:
                return StreamingResponse(_format_stream(summary), media_type="text/event-stream")

            return {
                "text": summary,
                "summary": summary,
                "word_count": len(summary.split()),
                "char_count": len(summary),
                "mode": mode,
                "section": section.strip() if section else "",
            }

        # ========== File Upload (Documents) ==========
        if files:
            temp_path = f"/tmp/{files.filename}"
            filename_lower = files.filename.lower()
            logger.info(f"Saving uploaded file: {files.filename}, type={type}")

            with open(temp_path, "wb") as buffer:
                content = await files.read()
                buffer.write(content)

            try:
                # PDF/DOC/DOCX
                if filename_lower.endswith((".pdf", ".docx", ".doc")):
                    file_type = "PDF" if filename_lower.endswith(".pdf") else "DOCX"
                    logger.info(f"Extracting text from {file_type} file")

                    text_content = pdf_service.extract_text(temp_path)
                    os.remove(temp_path)

                    if not text_content.strip():
                        raise HTTPException(status_code=400, detail=f"No text found in {file_type}")

                    # Create doc_id for caching
                    created_doc_id = llm_service.create_doc(text_content)

                    # Background prefetch for all sections
                    background_tasks.add_task(llm_service.prefetch_doc, created_doc_id)

                    # IMPORTANT FIX:
                    # If request is for a specific section, return that section (not initial summary)
                    if mode == "financial_section" and section.strip():
                        section_out = llm_service.summarize_by_doc_id(
                            doc_id=created_doc_id,
                            max_tokens=max_tokens,
                            stream=False,
                            mode="financial_section",
                            section=section.strip(),
                        )
                        return {
                            "doc_id": created_doc_id,
                            "text": section_out,
                            "summary": section_out,
                            "word_count": len(section_out.split()),
                            "char_count": len(section_out),
                            "mode": "financial_section",
                            "section": section.strip(),
                        }

                    # Otherwise return fast initial summary (4-5 sentences, first chunk only)
                    initial_summary = llm_service.initial_summary_first_chunk(created_doc_id)
                    return {
                        "doc_id": created_doc_id,
                        "text": initial_summary,
                        "summary": initial_summary,
                        "word_count": len(initial_summary.split()),
                        "char_count": len(initial_summary),
                        "mode": "financial_initial",
                        "section": "",
                    }

                # TXT
                if filename_lower.endswith(".txt"):
                    logger.info("Reading text from TXT file")
                    with open(temp_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    os.remove(temp_path)

                    if not text_content.strip():
                        raise HTTPException(status_code=400, detail="No text found in file")

                    created_doc_id = llm_service.create_doc(text_content)

                    background_tasks.add_task(llm_service.prefetch_doc, created_doc_id)

                    # Same fix for TXT:
                    if mode == "financial_section" and section.strip():
                        section_out = llm_service.summarize_by_doc_id(
                            doc_id=created_doc_id,
                            max_tokens=max_tokens,
                            stream=False,
                            mode="financial_section",
                            section=section.strip(),
                        )
                        return {
                            "doc_id": created_doc_id,
                            "text": section_out,
                            "summary": section_out,
                            "word_count": len(section_out.split()),
                            "char_count": len(section_out),
                            "mode": "financial_section",
                            "section": section.strip(),
                        }

                    initial_summary = llm_service.initial_summary_first_chunk(created_doc_id)
                    return {
                        "doc_id": created_doc_id,
                        "text": initial_summary,
                        "summary": initial_summary,
                        "word_count": len(initial_summary.split()),
                        "char_count": len(initial_summary),
                        "mode": "financial_initial",
                        "section": "",
                    }

                # Unsupported type
                logger.error(f"Unsupported file type: {files.filename}")
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Please upload PDF, DOCX, or TXT files.",
                )

            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

        # ========== Invalid Request ==========
        raise HTTPException(status_code=400, detail="Either text message or file is required")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


def _format_stream(generator):
    """Format streaming response for SSE"""
    try:
        for chunk in generator:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
