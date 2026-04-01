"""
API Routes for FinSights Application
Handles all HTTP endpoints
"""

from fastapi import APIRouter, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse
from typing import Optional
import os
import logging
import json

import config
from models import HealthResponse

from services import pdf_service, llm_service
from services.rag.rag_index_service import rag_index_service
from services.observability_service import observability_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_upload_warning(filename: str, document_info: dict) -> dict:
    over_size_limit = document_info["file_size_bytes"] > config.MAX_PDF_SIZE
    over_page_limit = bool(document_info.get("is_pdf")) and (document_info.get("page_count") or 0) > config.MAX_PDF_PAGES

    reasons = []
    if over_size_limit:
        reasons.append(
            f'"{filename}" is {document_info["file_size_mb"]} MB, above the {document_info["max_file_size_mb"]} MB limit.'
        )
    if over_page_limit:
        reasons.append(
            f'This PDF has {document_info["page_count"]} pages, and only the first {document_info["pages_to_process"]} pages will be processed.'
        )

    if not reasons:
        return {}

    summary = " ".join(reasons)
    if over_page_limit:
        summary += f" If you continue, FinSights will upload the file but use only the first {document_info['pages_to_process']} pages for extraction."
    elif over_size_limit:
        summary += " If you continue, FinSights will upload the full file and attempt normal extraction."

    return {
        "code": "upload_confirmation_required",
        "message": summary,
        "filename": filename,
        "requires_confirmation": True,
        "over_size_limit": over_size_limit,
        "over_page_limit": over_page_limit,
        "file_size_bytes": document_info["file_size_bytes"],
        "file_size_mb": document_info["file_size_mb"],
        "max_file_size_bytes": document_info["max_file_size_bytes"],
        "max_file_size_mb": document_info["max_file_size_mb"],
        "page_count": document_info.get("page_count"),
        "page_limit": document_info.get("page_limit"),
        "pages_to_process": document_info.get("pages_to_process"),
        "will_trim_pages": document_info.get("will_trim_pages", False),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    llm_health = llm_service.health_check()

    response = {
        "status": "healthy" if llm_health.get("status") == "healthy" else "unhealthy",
        "service": config.APP_TITLE,
        "version": config.APP_VERSION,
        "llm_provider": llm_health.get("provider", llm_service.get_provider_name()),
    }

    return response


@router.get("/v1/observability", response_class=PlainTextResponse)
async def observability(limit: int = 100):
    """
    Plain text table with LLM token observability only.
    """
    return observability_service.render_table(limit=limit, llm_only=True)

@router.get("/v1/rag/status")
async def rag_status(doc_id: str):
    """
    Returns RAG index status for a given doc_id.
    Frontend can poll this and enable Chat only when ready.
    """
    try:
        status = rag_index_service.get_status(doc_id.strip())
        return {"doc_id": doc_id.strip(), **status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/rag/chat")
async def rag_chat(
    doc_id: str = Form(""),
    message: str = Form(""),
    max_tokens: int = Form(220),
    temperature: float = Form(0.2),
):
    """
    RAG chat endpoint:
    - Retrieves top chunks from in-memory vector index for doc_id
    - Calls LLM using only retrieved context
    """
    try:
        doc_id_clean = (doc_id or "").strip()
        user_msg = (message or "").strip()

        if not doc_id_clean:
            raise HTTPException(status_code=400, detail="doc_id is required")
        if not user_msg:
            raise HTTPException(status_code=400, detail="message is required")

        # Ensure indexing is ready
        st = rag_index_service.get_status(doc_id_clean)
        if not st.get("ready"):
            return {
                "doc_id": doc_id_clean,
                "ready": False,
                "answer": "Indexing is still in progress. Please try again in a moment.",
                "retrieved_chunks": [],
            }

        # Retrieve context (top-k chunks)
        retrieved = rag_index_service.query(doc_id_clean, user_msg, top_k=4)
        context = "\n\n".join([r.get("text", "") for r in retrieved if r.get("text")])

        # Ask LLM with context only
        answer = llm_service.chat_with_context(
            question=user_msg,
            context=context,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

        return {
            "doc_id": doc_id_clean,
            "ready": True,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "chunk_id": r.get("chunk_id"),
                    "score": r.get("score"),
                }
                for r in retrieved
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG chat error: {str(e)}")

@router.delete("/v1/vectors/{doc_id}")
async def delete_vectors(doc_id: str):
    """
    Delete all vector embeddings for a given doc_id.
    Called when cleaning up before uploading a new document.
    """
    try:
        doc_id_clean = (doc_id or "").strip()
        if not doc_id_clean:
            raise HTTPException(status_code=400, detail="doc_id is required")
        
        from services.rag.vector_store import vector_store
        vector_store.clear_doc(doc_id_clean)
        
        return {"doc_id": doc_id_clean, "status": "deleted", "message": "Vector data cleared"}
    except Exception as e:
        logger.error(f"Vector deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector deletion error: {str(e)}")

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
    ignore_upload_warnings: str = Form("false"),
    files: Optional[UploadFile] = File(None),
):
    """
    Summarize text or document content (PDF/DOC/DOCX/TXT)

    Supports:
    - Cached path: doc_id present -> no need to resend file
    - File path: file present -> creates doc_id; if mode=financial_section returns that section immediately

    Dynamic sections:
    - For mode=financial_initial, response includes "sections" (2-5) when doc_id exists.
    - For text input, we also create a doc_id so we can return dynamic sections.
    """
    try:
        stream_bool = stream.lower() == "true"
        ignore_upload_warnings_bool = ignore_upload_warnings.lower() == "true"

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
        # User clicks on sections: frontend sends doc_id + mode=financial_section + section
        if doc_id.strip():
            doc_id_clean = doc_id.strip()
            try:
                summary = llm_service.summarize_by_doc_id(
                    doc_id=doc_id_clean,
                    max_tokens=max_tokens,
                    stream=stream_bool,
                    mode=mode,
                    section=section.strip() if section else None,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            if stream_bool:
                return StreamingResponse(_format_stream(summary), media_type="text/event-stream")

            resp = {
                "doc_id": doc_id_clean,
                "text": summary,
                "summary": summary,
                "word_count": len(summary.split()),
                "char_count": len(summary),
                "mode": mode,
                "section": section.strip() if section else "",
            }

            # Include dynamic sections for initial mode (and it is safe to include for any mode)
            try:
                resp["sections"] = llm_service.get_doc_sections(doc_id_clean)
            except Exception:
                resp["sections"] = []

            return resp

        # ========== Text Input ==========
        if type == "text" and messages.strip():
            logger.info("Processing text input")

            # For dynamic sections, create a doc_id for text as well
            # so frontend can reuse doc_id on chip clicks without resending text.
            created_doc_id = llm_service.create_doc(messages)

            # Background prefetch (no-op but keeps compatibility)
            background_tasks.add_task(llm_service.prefetch_doc, created_doc_id)

            # RAG indexing in background (does not block summary)
            background_tasks.add_task(rag_index_service.index_doc, created_doc_id)

            if mode == "financial_section" and section.strip():
                # If user explicitly requested a section on first call
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
                    "sections": llm_service.get_doc_sections(created_doc_id) or [],
                }

            if stream_bool and mode == "financial_overall":
                # Stream overall summary via doc_id path
                overall = llm_service.summarize_by_doc_id(
                    doc_id=created_doc_id,
                    max_tokens=max_tokens,
                    stream=True,
                    mode="financial_overall",
                    section=None,
                )
                return StreamingResponse(_format_stream(overall), media_type="text/event-stream")

            if mode == "financial_initial":
                initial_summary = llm_service.initial_summary_first_chunk(created_doc_id)
                sections = llm_service.get_doc_sections(created_doc_id) or []
                return {
                    "doc_id": created_doc_id,
                    "text": initial_summary,
                    "summary": initial_summary,
                    "word_count": len(initial_summary.split()),
                    "char_count": len(initial_summary),
                    "mode": "financial_initial",
                    "section": "",
                    "sections": sections,
                }

            # Other modes (non-stream)
            summary = llm_service.summarize_by_doc_id(
                doc_id=created_doc_id,
                max_tokens=max_tokens,
                stream=False,
                mode=mode,
                section=section.strip() if section else None,
            )
            return {
                "doc_id": created_doc_id,
                "text": summary,
                "summary": summary,
                "word_count": len(summary.split()),
                "char_count": len(summary),
                "mode": mode,
                "section": section.strip() if section else "",
                "sections": llm_service.get_doc_sections(created_doc_id) or [],
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

                    document_info = pdf_service.analyze_document(temp_path)
                    upload_warning = _build_upload_warning(files.filename, document_info)

                    if upload_warning and not ignore_upload_warnings_bool:
                        os.remove(temp_path)
                        raise HTTPException(status_code=413, detail=upload_warning)

                    text_content = pdf_service.extract_text(temp_path)
                    os.remove(temp_path)

                    if not text_content.strip():
                        raise HTTPException(status_code=400, detail=f"No text found in {file_type}")

                    # Create doc_id for caching
                    created_doc_id = llm_service.create_doc(text_content)

                    # Background prefetch (no-op but kept for compatibility)
                    background_tasks.add_task(llm_service.prefetch_doc, created_doc_id)

                    # RAG indexing in background (does not block summary)
                    background_tasks.add_task(rag_index_service.index_doc, created_doc_id)

                    # If request is for a specific section, return that section immediately
                    if mode == "financial_section" and section.strip():
                        section_out = llm_service.summarize_by_doc_id(
                            doc_id=created_doc_id,
                            max_tokens=max_tokens,
                            stream=False,
                            mode="financial_section",
                            section=section.strip(),
                        )
                        response = {
                            "doc_id": created_doc_id,
                            "text": section_out,
                            "summary": section_out,
                            "word_count": len(section_out.split()),
                            "char_count": len(section_out),
                            "mode": "financial_section",
                            "section": section.strip(),
                            "sections": llm_service.get_doc_sections(created_doc_id) or [],
                        }
                        if upload_warning:
                            response["upload_warning"] = upload_warning
                        return response

                    # Otherwise return fast initial summary (also discovers sections internally)
                    initial_summary = llm_service.initial_summary_first_chunk(created_doc_id)
                    sections = llm_service.get_doc_sections(created_doc_id) or []
                    response = {
                        "doc_id": created_doc_id,
                        "text": initial_summary,
                        "summary": initial_summary,
                        "word_count": len(initial_summary.split()),
                        "char_count": len(initial_summary),
                        "mode": "financial_initial",
                        "section": "",
                        "sections": sections,
                    }
                    if upload_warning:
                        response["upload_warning"] = upload_warning
                    return response

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

                    # RAG indexing in background (does not block summary)
                    background_tasks.add_task(rag_index_service.index_doc, created_doc_id)

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
                            "sections": llm_service.get_doc_sections(created_doc_id) or [],
                        }

                    initial_summary = llm_service.initial_summary_first_chunk(created_doc_id)
                    sections = llm_service.get_doc_sections(created_doc_id) or []
                    return {
                        "doc_id": created_doc_id,
                        "text": initial_summary,
                        "summary": initial_summary,
                        "word_count": len(initial_summary.split()),
                        "char_count": len(initial_summary),
                        "mode": "financial_initial",
                        "section": "",
                        "sections": sections,
                    }

                # Unsupported type
                logger.error(f"Unsupported file type: {files.filename}")
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Please upload PDF, DOCX, or TXT files.",
                )

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

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
