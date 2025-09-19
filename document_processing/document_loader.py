# document_processing/document_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple
import os
import re
from langchain.schema import Document

try:
    # pypdf is light and commonly available; good enough for headings/captions
    from pypdf import PdfReader
except ImportError:
    # fallback name on older installs
    from PyPDF2 import PdfReader  # type: ignore


_HEADING_PATTERNS = [
    re.compile(r"^\s*(section|chapter)\s+\d+(\.\d+)*\b[:.\- ]?", re.I),
    re.compile(r"^\s*\d+(\.\d+){0,3}\s+[-–:]?\s*[A-Z].{0,120}$"),  # 1. / 1.2.3 Title
    re.compile(r"^[A-Z][A-Z0-9 \-–:]{3,80}$"),                    # ALL CAPS line
]
_CAPTION_PATTERNS = [
    re.compile(r"^\s*table\s+\d+[:. ]", re.I),
    re.compile(r"^\s*figure\s+\d+[:. ]", re.I),
]
# Simple sentence splitter (avoid heavy deps)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 140:
        return False
    return any(p.search(s) for p in _HEADING_PATTERNS)

def _looks_like_caption(line: str) -> bool:
    s = line.strip()
    return any(p.search(s) for p in _CAPTION_PATTERNS)

def _normalize_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\u00AD", "")).strip()

def _estimate_tokens(text: str) -> int:
    # Light token estimator (≈4 chars/token). Works fine to control chunk size.
    # If you later add a tokenizer, replace this with exact counts.
    return max(1, int(len(text) / 4))

@dataclass
class ChunkingConfig:
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 36   # ~12%
    keep_captions_with_blocks: bool = True

class SemanticChunker:
    """
    Splits page text into semantic units:
      - Groups lines under detected headings
      - Attaches Table/Figure captions to nearest text block
      - Respects sentence boundaries while honoring token budget + overlap
    Produces langchain Documents with rich metadata.
    """

    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg

    def _group_blocks(self, lines: List[str]) -> List[Tuple[str, List[str]]]:
        """
        Returns list of (block_title, block_lines).
        A new block starts at a heading; captions are pulled into the current block.
        """
        blocks: List[Tuple[str, List[str]]] = []
        current_title = "Body"
        current_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = _normalize_space(lines[i])
            if not line:
                i += 1
                continue

            if _looks_like_heading(line):
                # flush previous block
                if current_lines:
                    blocks.append((current_title, current_lines))
                    current_lines = []
                current_title = line
                i += 1
                continue

            if _looks_like_caption(line) and self.cfg.keep_captions_with_blocks:
                # attach caption + possible next line (often wraps)
                caption = [line]
                if i + 1 < len(lines):
                    nxt = _normalize_space(lines[i + 1])
                    if nxt and not _looks_like_heading(nxt):
                        caption.append(nxt)
                        i += 1
                # prepend caption within the current block
                current_lines.extend(caption)
                i += 1
                continue

            current_lines.append(line)
            i += 1

        if current_lines:
            blocks.append((current_title, current_lines))

        return blocks

    def _sentences(self, text: str) -> List[str]:
        text = _normalize_space(text)
        if not text:
            return []
        # If the text is short or monolithic, just return as one sentence
        if len(text) < 240 or "." not in text:
            return [text]
        parts = _SENTENCE_SPLIT.split(text)
        # Merge very short fragments with neighbors
        merged: List[str] = []
        buf = ""
        for p in parts:
            if _estimate_tokens(p) < 10:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.append(p.strip())
        if buf:
            merged.append(buf)
        return merged

    def chunk_block(self, block_text: str, meta: Dict[str, Any]) -> Iterable[Document]:
        """
        Pack sentences into chunks with token budget and overlap.
        """
        sent = self._sentences(block_text)
        if not sent:
            return []

        size = self.cfg.chunk_size_tokens
        overlap = self.cfg.chunk_overlap_tokens
        window: List[str] = []
        window_tokens = 0
        idx = 0
        out: List[Document] = []

        def flush(doc_id: int, payload: List[str]):
            content = _normalize_space(" ".join(payload))
            if not content:
                return
            out.append(Document(page_content=content, metadata={**meta, "chunk_id": doc_id}))

        i = 0
        while i < len(sent):
            s = sent[i]
            t = _estimate_tokens(s)
            if window_tokens + t <= size or not window:
                window.append(s)
                window_tokens += t
                i += 1
            else:
                flush(idx, window)
                idx += 1
                # create overlap by taking sentences from the end until token budget ≈ overlap
                ov: List[str] = []
                ov_tokens = 0
                j = len(window) - 1
                while j >= 0 and ov_tokens < overlap:
                    ov.insert(0, window[j])
                    ov_tokens += _estimate_tokens(window[j])
                    j -= 1
                window = ov
                window_tokens = ov_tokens

        if window:
            flush(idx, window)

        return out

    def split_page(self, page_text: str, base_meta: Dict[str, Any]) -> List[Document]:
        # Split into non-empty lines first
        lines = [l for l in (page_text or "").splitlines()]
        blocks = self._group_blocks(lines)
        docs: List[Document] = []
        for title, blines in blocks:
            text = _normalize_space("\n".join(blines))
            meta = {**base_meta, "section_title": title}
            docs.extend(list(self.chunk_block(text, meta)))
        return docs


class DocumentLoader:
    """
    Public interface used by your RAG system.
    - process_pdf(path) -> List[Document]
    - get_processing_stats(docs) -> dict
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 36) -> None:
        self.cfg = ChunkingConfig(
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
        )
        self.chunker = SemanticChunker(self.cfg)

    def _read_pdf_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        reader = PdfReader(pdf_path)
        pages: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append((i, txt))
        return pages

    def process_pdf(self, pdf_path: str) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        basename = os.path.basename(pdf_path)
        all_docs: List[Document] = []
        for page_no, text in self._read_pdf_pages(pdf_path):
            base_meta = {
                "page": page_no,
                "document_name": basename,
                "source_pdf": pdf_path,
            }
            page_docs = self.chunker.split_page(text, base_meta)
            all_docs.extend(page_docs)

        # Attach a simple running chunk index per page to keep IDs stable/readable
        by_page: Dict[int, int] = {}
        for d in all_docs:
            p = d.metadata.get("page", -1)
            by_page[p] = by_page.get(p, 0) + 1
            d.metadata["chunk_idx_on_page"] = by_page[p]

        return all_docs

    def get_processing_stats(self, docs: List[Document]) -> Dict[str, Any]:
        if not docs:
            return {"count": 0, "avg_chunk_size": 0, "pages": 0}
        sizes = [len(d.page_content) for d in docs]
        pages = {d.metadata.get("page") for d in docs}
        return {
            "count": len(docs),
            "avg_chunk_size": int(sum(sizes) / len(sizes)),
            "pages": len(pages),
        }
