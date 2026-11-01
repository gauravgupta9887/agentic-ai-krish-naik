"""
Load the corpus into plain page-level records.

Two ingestion lessons are baked in here (both bit us on this very corpus):

1. NFKC normalisation. PDF text extraction returns typographic ligatures such as 'ﬁ' and
   'ﬂ' as single code points, so "definition" arrives as "deﬁnition" and never matches a
   user's query or a ground-truth string. NFKC folds them back to plain letters.
2. Empty-text PDFs. git.pdf is image-only: pypdf returns 0 characters. A pipeline that
   does not check this "succeeds" while silently indexing nothing. We log and skip it.
"""
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from config import CORPUS_DIR


@dataclass
class Page:
    text: str
    source: str      # file name
    page: int        # 1-based page number (1 for .txt)


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("­", "").strip()


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix == ".pdf":
            for i, p in enumerate(PdfReader(str(path)).pages, start=1):
                text = normalise(p.extract_text() or "")
                if text:
                    pages.append(Page(text, path.name, i))
        elif path.suffix == ".txt":
            text = normalise(path.read_text(encoding="utf-8"))
            if text:
                pages.append(Page(text, path.name, 1))
    return pages
