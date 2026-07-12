from __future__ import annotations
import re
from rag.config import DOCS_ROOT, EXCLUDE_FILES
from dataclasses import dataclass
from pathlib import Path


@dataclass 
class Document:
    title: str
    text: str
    path: str

def clean_markdown(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)
    text = re.sub(r"\{\*.*?\*\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{!.*?!\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text

def parse_document(path: Path, docs_root=DOCS_ROOT) -> Document | None:
    raw = path.read_text(encoding="utf-8")
    text = clean_markdown(raw)
    
    if len(text.strip()) < 200:
        return None
    
    return Document(
            title=path.stem,
            text=text,
            path=str(path.relative_to(docs_root)),
        )
    
def load_documents(docs_root: Path = DOCS_ROOT) -> list[Document]:
    docs = []
    for p in docs_root.rglob("*.md"):
        
        if p.name in EXCLUDE_FILES:
            continue
        
        doc = parse_document(p, docs_root)
        if doc is not None:
            docs.append(doc)
            
    return docs
