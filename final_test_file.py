import os
import sys
import unittest
import textwrap

# Ensure the backend path is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from chatbot import TextExtractorDOCX, FileObject, TextChunk

class TestTextExtractorDOCX(unittest.TestCase):
    def setUp(self):
        # Path to our sample .docx in the test_files directory
        self.test_dir = os.path.dirname(__file__)
        self.file_path = os.path.join(self.test_dir, 'test_files', 'sample_file.docx')
        self.file = FileObject(url=self.file_path)

    def test_supports_docx_and_rejects_other(self):
        extractor = TextExtractorDOCX()
        self.assertTrue(extractor.supports(self.file_path))
        self.assertFalse(extractor.supports('document.txt'))

    def test_extract_from_sample_file(self):
        extractor = TextExtractorDOCX(markdown=True)
        chunks = extractor.extract(self.file)

        # Should return a single TextChunk
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 1)

        # Compare the extracted markdown to expected content
        extracted_text = chunks[0].text.strip()
        expected_text = textwrap.dedent("""
            ## Key concepts and entities

            The FIS Risk Trade Store service provides a store of AA-format trades supporting multiple versions and audit history of each change. It also supports back-dates.
        """).strip()

        self.assertEqual(extracted_text, expected_text)

if __name__ == '__main__':
    unittest.main()
