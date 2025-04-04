import time
from typing import Union
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

    @staticmethod
    def compare_performance(file_path: str):
        """
        Measure performance of mammoth loader.

        :param file_path: Path to the .docx file.
        """
        start_time = time.time()
        mammoth_content = DocxLoaderComparison.from_mammoth(file_path)
        mammoth_time = time.time() - start_time

        print(f"\nMammoth Loader Time: {mammoth_time:.4f} seconds")

    @staticmethod
    def evaluate_extraction(extracted_text: str, reference_text: str) -> float:
        """
        Evaluate extraction accuracy using simple token overlap method.

        :param extracted_text: Text extracted by the loader.
        :param reference_text: Ground truth text for comparison.
        :return: Similarity score (percentage).
        """
        extracted_tokens = set(extracted_text.split())
        reference_tokens = set(reference_text.split())
        intersection = extracted_tokens.intersection(reference_tokens)
        similarity = len(intersection) / max(len(extracted_tokens), len(reference_tokens)) * 100

        return similarity

    @staticmethod
    def run_evaluation(file_path: str, reference_text: str):
        """
        Run performance and accuracy evaluation on a given .docx file.

        :param file_path: Path to the .docx file.
        :param reference_text: Reference text for accuracy comparison.
        """
        # Performance Check
        DocxLoaderComparison.compare_performance(file_path)

        # Extraction
        extracted_text = DocxLoaderComparison.from_mammoth(file_path)

        # Accuracy Check
        similarity_score = DocxLoaderComparison.evaluate_extraction(extracted_text, reference_text)
        print(f"\nSimilarity Score: {similarity_score:.2f}%")
