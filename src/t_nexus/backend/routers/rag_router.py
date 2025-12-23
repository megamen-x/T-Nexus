"""
RAG-backed endpoints used by the Telegram bot.
"""

from __future__ import annotations

import json
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.t_nexus.ml.hipporag import ChatHistory, HippoRAG
from src.t_nexus.ml.utils.document_converter import collect_texts


CONFIG_PATH = Path(__file__).resolve().parents[2] / "ml" / "config" / "hipporag.yaml"
_hipporag = HippoRAG(str(CONFIG_PATH))

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    question: List[str]


class RAGResponse(BaseModel):
    full_answer: List[str]
    short_answer: List[str]
    docs: List[List[str]]


@router.get("/get_database_name/")
def get_database_name() -> dict:
    return {"message": _hipporag.state_dir}


@router.post("/index/")
async def index_files(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        temp_zip.write(content)
        temp_zip.flush()
        temp_zip.close()

        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            documents = collect_texts([Path(extract_dir)])
            if not documents:
                return {"message": "No documents found inside the archive."}

            chunk_ids = await _hipporag.index_documents(documents)
    finally:
        try:
            os.unlink(temp_zip.name)
        except OSError:
            pass

    return {"message": f"Indexed {len(chunk_ids)} chunks."}


@router.post("/request_processing/", response_model=RAGResponse)
async def request_processing(request: RAGRequest) -> RAGResponse:
    if not request.question:
        raise HTTPException(status_code=400, detail="Empty question list is not allowed.")

    full_answers: List[str] = []
    short_answers: List[str] = []
    docs: List[List[str]] = []

    for question in request.question:
        history = ChatHistory()
        history.add("user", question)
        answer, retrieval = await _hipporag.rag(history)
        answer = json.loads(answer)
        full_answers.append(answer['full_answer'])
        short_answers.append(answer['short_answer'])

        sources = [
            p.source for p in retrieval.passages if p.source
        ]
        docs.append(sources)

    return RAGResponse(
        full_answer=full_answers,
        short_answer=short_answers,
        docs=docs,
    )
