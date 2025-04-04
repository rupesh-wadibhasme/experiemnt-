import os
import time
from typing import Union
import mammoth
import base64
import re


class DocxLoaderWithArtifacts:

    @classmethod
    def from_mammoth(cls, file_path: str, save_markdown: bool = True, save_artifacts: bool = True, artifacts_folder: str = "artifacts") -> str:
        """
        Load a .docx file using Mammoth, save markdown text, and save artifacts.

        :param file_path: Path to the .docx file.
        :param save_markdown: If True, saves the extracted text as a Markdown file.
        :param save_artifacts: If True, saves extracted artifacts (images) to a folder.
        :param artifacts_folder: Directory to save extracted artifacts.
        :return: Extracted text as a string (Markdown format).
        """
        # Create artifacts directory if not exists
        if save_artifacts and not os.path.exists(artifacts_folder):
            os.makedirs(artifacts_folder)
        
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
            extracted_text = result.value  # Returns markdown-formatted text
            
            # Extract base64 encoded images using regex
            artifact_count = 0
            artifacts = re.findall(r'!\[.*?\]\(data:image\/.*?;base64,(.*?)\)', extracted_text)

            for artifact in artifacts:
                artifact_count += 1
                # Save image as file
                artifact_data = base64.b64decode(artifact)
                artifact_file_path = os.path.join(artifacts_folder, f"image_{artifact_count}.png")
                
                with open(artifact_file_path, "wb") as f:
                    f.write(artifact_data)

                print(f"Saved artifact: {artifact_file_path}")

            # Optionally save the markdown file
            if save_markdown:
                with open("mammoth_output.md", "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print("Mammoth output saved as 'mammoth_output.md'")
        
        return extracted_text

    @staticmethod
    def run_evaluation(file_path: str, save_markdown: bool = True, save_artifacts: bool = True):
        """
        Run extraction and save artifacts.

        :param file_path: Path to the .docx file.
        :param save_markdown: If True, saves the extracted text as a Markdown file.
        :param save_artifacts: If True, saves extracted artifacts (images) to a folder.
        """
        # Extract text and artifacts using mammoth
        extracted_text = DocxLoaderWithArtifacts.from_mammoth(
            file_path,
            save_markdown=save_markdown,
            save_artifacts=save_artifacts
        )

        print(f"\nLength of extracted text: {len(extracted_text)} characters")
