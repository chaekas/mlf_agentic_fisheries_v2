"""Universal document ingestion for the MLF Fisheries Agentic AI.

Supported files:
PDF, DOCX, XLSX, CSV, PPTX, TXT, MD, JSON, XML, HTML and common images.
Images use OCR when Tesseract is installed on the computer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from PIL import Image
from pptx import Presentation
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

try:
    import pytesseract
except ImportError:
    pytesseract = None

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5433/mlf_fisheries",
)
COLLECTION = os.getenv("PGVECTOR_COLLECTION", "mlf_fisheries_regulations")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".pptx",
    ".txt", ".md", ".json", ".xml", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def detect_lang(text: str) -> str:
    try:
        result = detect(text[:1500])
        return "sw" if result.startswith("sw") else "en"
    except Exception:
        return "en"


def text_document(text: str, source: str, metadata: dict | None = None) -> List[Document]:
    text = (text or "").strip()
    if not text:
        return []

    base = {
        "source": source,
        "lang": detect_lang(text),
    }
    if metadata:
        base.update(metadata)

    return [
        Document(page_content=chunk, metadata=base.copy())
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]


def load_pdf(path: Path) -> List[Document]:
    docs: List[Document] = []
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        docs.extend(text_document(text, path.name, {"page": page_no, "file_type": "pdf"}))
    return docs


def load_docx(path: Path) -> List[Document]:
    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    table_rows = []
    for table in document.tables:
        for row in table.rows:
            table_rows.append(" | ".join(cell.text.strip() for cell in row.cells))
    text = "\n".join(paragraphs + table_rows)
    return text_document(text, path.name, {"file_type": "docx"})


def load_spreadsheet(path: Path) -> List[Document]:
    sheets = pd.read_excel(path, sheet_name=None)
    docs: List[Document] = []
    for sheet_name, frame in sheets.items():
        frame = frame.fillna("")
        text = f"Sheet: {sheet_name}\n{frame.to_csv(index=False)}"
        docs.extend(text_document(text, path.name, {"file_type": "xlsx", "sheet": str(sheet_name)}))
    return docs


def load_csv(path: Path) -> List[Document]:
    frame = pd.read_csv(path).fillna("")
    return text_document(frame.to_csv(index=False), path.name, {"file_type": "csv"})


def load_pptx(path: Path) -> List[Document]:
    presentation = Presentation(str(path))
    docs: List[Document] = []
    for slide_no, slide in enumerate(presentation.slides, 1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        docs.extend(text_document("\n".join(texts), path.name, {"file_type": "pptx", "slide": slide_no}))
    return docs


def load_plain(path: Path) -> List[Document]:
    return text_document(path.read_text(encoding="utf-8", errors="ignore"), path.name, {"file_type": path.suffix.lower().lstrip(".")})


def load_json(path: Path) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return text_document(json.dumps(data, ensure_ascii=False, indent=2), path.name, {"file_type": "json"})


def load_image(path: Path) -> List[Document]:
    if pytesseract is None:
        print(f"WARNING: OCR package unavailable; skipped image: {path.name}")
        return []
    try:
        text = pytesseract.image_to_string(Image.open(path))
        return text_document(text, path.name, {"file_type": "image", "ocr": True})
    except Exception as exc:
        print(f"WARNING: OCR failed for {path.name}: {exc}")
        return []


def load_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in {".xlsx", ".xlsm"}:
        return load_spreadsheet(path)
    if ext == ".csv":
        return load_csv(path)
    if ext == ".pptx":
        return load_pptx(path)
    if ext in {".txt", ".md", ".xml", ".html", ".htm"}:
        return load_plain(path)
    if ext == ".json":
        return load_json(path)
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
        return load_image(path)
    return []


def collect_paths(items: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
        else:
            print(f"Skipping unsupported or missing path: {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Index MLF Fisheries documents into PostgreSQL/pgvector.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    args = parser.parse_args()

    paths = collect_paths(args.paths)
    if not paths:
        print("No supported files found.")
        return

    docs: List[Document] = []
    for path in paths:
        print(f"Reading: {path}")
        try:
            loaded = load_file(path)
            docs.extend(loaded)
            print(f"  -> {len(loaded)} chunks")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not docs:
        print("No text could be extracted from the supplied files.")
        return

    print(f"Creating embeddings with {EMBEDDING_MODEL}...")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    store = PGVector(
        embeddings=emb,
        collection_name=COLLECTION,
        connection=DATABASE_URL,
        use_jsonb=True,
        create_extension=True,
    )
    store.add_documents(docs)

    print(f"\nSUCCESS: indexed {len(docs)} chunks.")
    print(f"Collection: {COLLECTION}")
    print(f"Database: {DATABASE_URL.split('@')[-1]}")


if __name__ == "__main__":
    main()
