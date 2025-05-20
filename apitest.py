import os
import sys
import unittest
from unittest.mock import patch
from app import app

class SimpleAPITestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    @patch("backend.service_api.get_all_document_metrics")
    def test_document_metrics(self, mock_get):
        mock_get.return_value = [{"id": "doc1"}]
        response = self.client.get("/document-metrics")
        self.assertEqual(response.status_code, 200)

    @patch("backend.service_api.get_document_metrics_by_name")
    def test_document_metrics_by_name(self, mock_get):
        mock_get.return_value = [{"id": "doc2"}]
        response = self.client.get("/document-metrics?name=testdoc")
        self.assertEqual(response.status_code, 200)

    @patch("backend.service_api.get_all_chat_sessions")
    @patch("backend.service_api.summarize_chat_sessions")
    def test_chat_metrics(self, mock_summary, mock_get):
        mock_get.return_value = [{"session_id": "s1"}]
        mock_summary.return_value = {"average": 1}
        response = self.client.get("/chat-metrics")
        self.assertEqual(response.status_code, 200)

    @patch("backend.service_api.add_feedback")
    def test_feedback(self, mock_add):
        response = self.client.post("/feedback", json={
            "session_id": "s1",
            "answer_id": "a1",
            "feedback": "good"
        })
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
