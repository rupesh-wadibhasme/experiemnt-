def test_docx_file(self):
    file_path = os.path.join(self.test_dir, "sample_file.docx")
    result = TextLoader.from_file(file_path)

    # strip out newlines so that small formatting differences don’t break the assertion
    actual = result[0].text.replace("\n", "")
    expected = "## Key concepts and entitiesThe FIS Risk Trade Store service provides a store of AA-format trades supporting multiple versions and audit history of each change. It also supports back-dates…"

    self.assertEqual(actual, expected)
