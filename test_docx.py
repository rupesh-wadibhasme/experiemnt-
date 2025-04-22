import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# ensure backend path is on sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from chatbot import TextExtractorDOCX, FileObject, TextChunk

class TestTextExtractorDOCX(unittest.TestCase):
    def setUp(self):
        # Simulate a .docx file URL
        self.file_path = "/path/to/test_document.docx"
        self.file = FileObject(url=self.file_path)

    def test_supports_docx_and_rejects_other(self):
        extractor = TextExtractorDOCX()
        self.assertTrue(extractor.supports(self.file_path))
        self.assertFalse(extractor.supports("file.pdf"))

    @patch('chatbot.Document')
    def test_extract__loads_paragraphs_and_formats(self, MockDocument):
        # Prepare two mocked paragraphs:
        # 1) a normal paragraph
        mock_run1 = MagicMock(text="Hello world", bold=False, italic=False)
        mock_para1 = MagicMock(runs=[mock_run1], style=MagicMock(name="Normal"))

        # 2) a heading paragraph (Heading 2)
        mock_run2 = MagicMock(text="Section Title", bold=False, italic=False)
        mock_style2 = MagicMock(name="Heading 2")
        mock_para2 = MagicMock(runs=[mock_run2], style=mock_style2)

        # The Document(...) call should return an object with our paragraphs
        MockDocument.return_value = MagicMock(paragraphs=[mock_para1, mock_para2])

        extractor = TextExtractorDOCX(markdown=True)
        chunks = extractor.extract(self.file)

        # We should get exactly one TextChunk back
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 1)

        # Extracted markdown should contain both paragraphs,
        # with Heading 2 rendered as "## Section Title"
        full_text = chunks[0].text
        self.assertIn("Hello world", full_text)
        self.assertIn("## Section Title", full_text)

        # Ensure python-docx Document was called with our file path
        MockDocument.assert_called_once_with(self.file_path)

if __name__ == '__main__':
    unittest.main()
