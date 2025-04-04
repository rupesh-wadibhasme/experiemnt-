import time
from typing import Union
from docx import Document as DocxDocument
from langchain.document_loaders import DocxLoader
from langchain.schema import Document
import tempfile
import mammoth


class DocxLoaderComparison:

    @classmethod
    def from_langchain_loader(cls, file_path: str) -> list[Document]:
        """
        Load a .docx file using LangChain's DocxLoader.

        :param file_path: Path to the .docx file.
        :return: List of LangChain Document objects.
        """
        loader = DocxLoader(file_path)
        return loader.load()

    @classmethod
    def from_custom_loader(cls, file_path: str, markdown: bool = True) -> str:
        """
        Load a .docx file using custom loader with python-docx and mammoth.

        :param file_path: Path to the .docx file.
        :param markdown: Whether to extract markdown using mammoth or plain text.
        :return: Extracted text as a string.
        """
        if markdown:
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_markdown(docx_file)
                return result.value  # Returns markdown-formatted text

        # Using python-docx for text extraction
        doc = DocxDocument(file_path)
        text_content = []

        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                text_content.append(f"# {para.text}")  # Convert headings to Markdown style
            else:
                text_content.append(para.text)

        return "\n\n".join(text_content)

    @staticmethod
    def compare_performance(file_path: str):
        """
        Compare the performance of the LangChain loader vs. the custom loader.

        :param file_path: Path to the .docx file.
        """
        # Measure performance of LangChain Loader
        start_time = time.time()
        langchain_docs = DocxLoaderComparison.from_langchain_loader(file_path)
        langchain_time = time.time() - start_time

        # Measure performance of Custom Loader
        start_time = time.time()
        custom_content = DocxLoaderComparison.from_custom_loader(file_path)
        custom_time = time.time() - start_time

        print(f"\nLangChain Loader Time: {langchain_time:.4f} seconds")
        print(f"Custom Loader Time: {custom_time:.4f} seconds")

    @staticmethod
    def compare_accuracy(file_path: str):
        """
        Compare the accuracy of LangChain loader and custom loader.

        :param file_path: Path to the .docx file.
        """
        # Extract text using both methods
        langchain_docs = DocxLoaderComparison.from_langchain_loader(file_path)
        custom_content = DocxLoaderComparison.from_custom_loader(file_path)

        # Combine all LangChain loader content
        langchain_content = "\n\n".join([doc.page_content for doc in langchain_docs])

        print("\nLangChain Loader Output:")
        print(langchain_content[:500])  # Print first 500 characters

        print("\nCustom Loader Output:")
        print(custom_content[:500])  # Print first 500 characters

        # Check similarity (simple token match)
        langchain_tokens = set(langchain_content.split())
        custom_tokens = set(custom_content.split())
        intersection = langchain_tokens.intersection(custom_tokens)
        similarity = len(intersection) / max(len(langchain_tokens), len(custom_tokens)) * 100

        print(f"\nSimilarity Score: {similarity:.2f}%")

