import logging
import os
import re
from dataclasses import dataclass, field

import pymupdf

from rag.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: int
    text: str
    parent_id: int
    page_num: int
    source_file: str


@dataclass
class ParentChunk:
    id: int
    text: str
    page_num: int
    source_file: str


def _extract_text_from_pdf(filepath: str) -> list[tuple[int, str]]:
    pages = []
    doc = pymupdf.open(filepath)
    filename = os.path.basename(filepath)
    for page_num in range(min(len(doc), settings.MAX_PAGES)):
        text = doc[page_num].get_text().strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    logger.info("Extracted %d pages from %s", len(pages), filename)
    return pages


def _split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if len(p.strip()) > 20]


def _split_into_small_chunks(paragraph: str, max_size: int = 300) -> list[str]:
    if len(paragraph) <= max_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_size:
            current += " " + sent if current else sent
        else:
            if current:
                chunks.append(current.strip())
            if len(sent) <= max_size:
                current = sent
            else:
                for i in range(0, len(sent), max_size):
                    chunks.append(sent[i : i + max_size].strip())
                current = ""
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 10]


def load_and_chunk_documents() -> tuple[list[Chunk], list[ParentChunk]]:
    artifacts_dir = settings.ARTIFACTS_DIR
    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    pdf_files = [f for f in os.listdir(artifacts_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {artifacts_dir}")

    chunks: list[Chunk] = []
    parent_chunks: list[ParentChunk] = []
    chunk_id = 0
    parent_id = 0

    for pdf_file in pdf_files:
        filepath = os.path.join(artifacts_dir, pdf_file)
        pages = _extract_text_from_pdf(filepath)

        for page_num, text in pages:
            paragraphs = _split_into_paragraphs(text)
            for para in paragraphs:
                parent = ParentChunk(
                    id=parent_id,
                    text=para,
                    page_num=page_num,
                    source_file=pdf_file,
                )
                parent_chunks.append(parent)

                small_parts = _split_into_small_chunks(para, settings.SMALL_CHUNK_SIZE)
                for part in small_parts:
                    child = Chunk(
                        id=chunk_id,
                        text=part,
                        parent_id=parent_id,
                        page_num=page_num,
                        source_file=pdf_file,
                    )
                    chunks.append(child)
                    chunk_id += 1

                parent_id += 1

    logger.info(
        "Created %d small chunks from %d parent chunks (%d PDFs)",
        len(chunks), len(parent_chunks), len(pdf_files),
    )
    return chunks, parent_chunks
