"""
Document Loader and Text Chunker for ICAR and KVK Agricultural Guidelines (Week 6).
Parses markdown and text advisory publications into clean, metadata-tagged chunks.
"""

from pathlib import Path
from typing import List, Dict, Any


class DocumentChunk:
    """Represents a coherent section of an agricultural guideline document."""

    def __init__(self, content: str, source: str, section: str):
        self.content = content.strip()
        self.source = source
        self.section = section

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "section": self.section
        }


class AgriculturalDocumentLoader:
    """Loads and splits agricultural advisory publications into semantic chunks."""

    def __init__(self, docs_dir: Path | str | None = None):
        """Initialize loader with directory containing markdown/text manuals."""
        if docs_dir is None:
            self.docs_dir = Path(__file__).resolve().parent.parent.parent / "docs"
        else:
            self.docs_dir = Path(docs_dir)

    def load_guidelines(self) -> List[DocumentChunk]:
        """
        Scan directory for markdown files and segment them into contextual chunks.
        """
        chunks: List[DocumentChunk] = []

        if not self.docs_dir.exists():
            return chunks

        md_files = list(self.docs_dir.glob("*.md")) + list(self.docs_dir.glob("*.txt"))

        for file_path in md_files:
            try:
                text = file_path.read_text(encoding="utf-8")
                # Split along major markdown sections (##)
                raw_sections = text.split("\n## ")

                # First chunk might be document title
                doc_title = raw_sections[0].strip().replace("# ", "")

                for idx, section in enumerate(raw_sections[1:], start=1):
                    lines = section.split("\n", 1)
                    section_title = lines[0].strip()
                    section_body = lines[1].strip() if len(lines) > 1 else section_title

                    # Build high-density semantic text
                    full_chunk = f"Section: {section_title}\n{section_body}"
                    chunks.append(DocumentChunk(
                        content=full_chunk,
                        source=file_path.name,
                        section=section_title
                    ))
            except Exception:
                continue

        return chunks
