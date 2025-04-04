import time
from typing import Union
from docx import Document as DocxDocument
import mammoth


class DocxLoaderComparison:

    @classmethod
    def from_mammoth(cls, file_path: str) -> str:
        """
        Load a .docx file using Mammoth.

        :param file_path: Path to the .docx file.
        :return: Extracted text as a string (Markdown format).
        """
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
            return result.value  # Returns markdown-formatted text

    @classmethod
    def from_python_docx(cls, file_path: str) -> str:
        """
        Load a .docx file using python-docx.

        :param file_path: Path to the .docx file.
        :return: Extracted text as a string.
        """
        doc = DocxDocument(file_path)
        extracted_text = []

        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                extracted_text.append(f"# {para.text}")
            else:
                extracted_text.append(para.text)

        return "\n\n".join(extracted_text)

    @staticmethod
    def compare_performance(file_path: str):
        """
        Measure performance of both loaders.

        :param file_path: Path to the .docx file.
        """
        # Mammoth Performance
        start_time = time.time()
        mammoth_content = DocxLoaderComparison.from_mammoth(file_path)
        mammoth_time = time.time() - start_time

        # python-docx Performance
        start_time = time.time()
        python_docx_content = DocxLoaderComparison.from_python_docx(file_path)
        python_docx_time = time.time() - start_time

        print(f"\nMammoth Loader Time: {mammoth_time:.4f} seconds")
        print(f"Python-docx Loader Time: {python_docx_time:.4f} seconds")

    @staticmethod
    def evaluate_extraction(mammoth_text: str, python_docx_text: str) -> float:
        """
        Evaluate extraction accuracy using simple token overlap method.

        :param mammoth_text: Text extracted by the Mammoth loader.
        :param python_docx_text: Text extracted by the python-docx loader.
        :return: Similarity score (percentage).
        """
        mammoth_tokens = set(mammoth_text.split())
        python_docx_tokens = set(python_docx_text.split())
        intersection = mammoth_tokens.intersection(python_docx_tokens)
        similarity = len(intersection) / max(len(mammoth_tokens), len(python_docx_tokens)) * 100

        return similarity

    @staticmethod
    def run_evaluation(file_path: str):
        """
        Run performance and accuracy evaluation on a given .docx file.

        :param file_path: Path to the .docx file.
        """
        # Extract text using both methods
        mammoth_content = DocxLoaderComparison.from_mammoth(file_path)
        python_docx_content = DocxLoaderComparison.from_python_docx(file_path)

        # Compare Performance
        DocxLoaderComparison.compare_performance(file_path)

        # Compare Accuracy
        similarity_score = DocxLoaderComparison.evaluate_extraction(mammoth_content, python_docx_content)
        print(f"\nSimilarity Score between Mammoth and python-docx: {similarity_score:.2f}%")
