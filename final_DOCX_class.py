import re
from docx import Document
from typing import List
from chatbot import TextExtractor, FileObject, TextChunk

class TextExtractorDOCX(TextExtractor):
    """Extracts text from .docx files in Markdown format, preserving headings and run formatting."""
    def __init__(self, markdown: bool = True):
        # If markdown=True, wraps bold/italic runs and headings in Markdown syntax.
        self.markdown = markdown

    def supports(self, url: str) -> bool:
        # Only handle .docx files
        return url.lower().endswith('.docx')

    def extract(self, file: FileObject) -> List[TextChunk]:
        # Inner loader that reads the file path and returns TextChunk(s)
        def load(path: str) -> List[TextChunk]:
            doc = Document(path)
            lines: List[str] = []

            def get_content(runs) -> str:
                segments = []
                for run in runs:
                    text = run.text or ''
                    if self.markdown:
                        if run.bold:
                            text = f"**{text}**"
                        if run.italic:
                            text = f"*{text}*"
                    segments.append(text)
                return ''.join(segments).strip()

            def heading_level(style_name: str) -> int:
                # Matches style names like "Heading 1", "Heading 2", etc.
                match = re.search(r"Heading\s*(\d+)", style_name, re.IGNORECASE)
                return int(match.group(1)) if match else 1

            for paragraph in doc.paragraphs:
                content = get_content(paragraph.runs)
                if not content:
                    continue

                style_name = (paragraph.style.name or '').lower()
                if self.markdown and style_name.startswith('heading'):
                    level = min(heading_level(paragraph.style.name), 6)
                    lines.append(f"{'#' * level} {content}")
                else:
                    lines.append(content)

            markdown_text = '\n\n'.join(lines)
            return [TextChunk(markdown_text, file.url, file.name)]

        # Directly invoke our loader on file.url (bypassing FileObject.path)
        return load(file.url)
