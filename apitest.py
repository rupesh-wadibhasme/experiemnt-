import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from main import app

class FlaskAppMockedTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch('service_api.get_all_documents')  # ✅ Match your import style
    def test_analytics_mocked(self, mock_get):
        mock_get.return_value = [{"id": "doc1"}]
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, [{"id": "doc1"}])

    @patch('service_api.get_all_chat_sessions')
    @patch('service_api.summarize_chat_sessions')
    def test_chat_metrics_mocked(self, mock_summary, mock_get):
        mock_get.return_value = [{"id": "chat1"}]
        mock_summary.return_value = {"average": 1}
        response = self.client.get("/chat-metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("summary", response.json)
        self.assertIn("data", response.json)

if __name__ == '__main__':
    unittest.main()
