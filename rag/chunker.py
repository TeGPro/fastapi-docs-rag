from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.data_loader import Document
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: int
    doc_path: str
    heading: str
    text: str
    
def split_into_sections(doc: Document) -> list[tuple[str, str]]:
    in_code = False
    lines = doc.text.split("\n")
    sections = []
    current_lines = []
    current_heading = doc.title
    
    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_heading, text))
    
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            current_lines.append(line)
            continue
        if not in_code and line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = []
            continue
        
        current_lines.append(line)
    
    flush()
            
    return sections

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    assert 0 <= overlap < chunk_size
    step = chunk_size - overlap
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        window = words[start: start + chunk_size]
        if len(window) <= overlap and chunks:
            break
            
        chunks.append(" ".join(window))
        start += step
        
    return chunks

def chunk_documents(docs: list, chunk_size: int, overlap: int) -> list[Chunk]:
    chunk_id = 0
    result = []
    
    for doc in docs:
        sections = split_into_sections(doc)
        for (heading, section_text) in sections:
            pieces = chunk_text(section_text, chunk_size, overlap)
            for piece in pieces:
                prefixed = f"{doc.title} / {heading}\n\n{piece}"
                result.append(Chunk(
                    chunk_id=chunk_id,
                    doc_path=doc.path,
                    heading=heading,
                    text=prefixed
                ))
                
                chunk_id += 1
    return result