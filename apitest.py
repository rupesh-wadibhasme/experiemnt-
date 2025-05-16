# test_monitoring_api.py

import unittest
from unittest.mock import patch, MagicMock
from monitoring_api import app

class FlaskAppMockedTestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    @patch("monitoring_api.load_config")
    @patch("monitoring_api.CosmosClient")
    def test_analytics_mocked(self, mock_cosmos, mock_config):
        # Mock config secrets
        mock_config.return_value = {
            "DB_URL": "fake-url",
            "DB_KEY": "fake-key",
            "DB_NAME": "fake-db"
        }

        # Mock Cosmos container
        mock_container = MagicMock()
        mock_container.read_all_items.return_value = [
            {
                "id": "doc1",
                "timestamp": "2025-04-01T10:00:00",
                "chunks": 3,
                "tokens": 300,
            }
        ]

        # Mock DB setup
        mock_db = MagicMock()
        mock_db.get_container_client.return_value = mock_container
        mock_cosmos.return_value.get_database_client.return_value = mock_db

        # Call API
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("doc1" in str(response.data))

    @patch("monitoring_api.load_config")
    @patch("monitoring_api.CosmosClient")
    def test_chat_metrics_mocked(self, mock_cosmos, mock_config):
        # Mock config
        mock_config.return_value = {
            "DB_URL": "fake-url",
            "DB_KEY": "fake-key",
            "DB_NAME": "fake-db"
        }

        # Mock container with multiple sessions and feedback
        mock_container = MagicMock()
        mock_container.read_all_items.return_value = [
            {
                "service_id": "PROPHET",
                "session_id": "abc123",
                "response_time": 6.2,
                "feedback": "good"
            },
            {
                "service_id": "PROPHET",
                "session_id": "abc123",
                "response_time": 5.8,
                "feedback": "neutral"
            }
        ]

        # Cosmos mocking
        mock_db = MagicMock()
        mock_db.get_container_client.return_value = mock_container
        mock_cosmos.return_value.get_database_client.return_value = mock_db

        response = self.client.get("/chat-metrics")
        self.assertEqual(response.status_code, 200)

        json_data = response.get_json()
        self.assertIn("summary", json_data)
        self.assertIn("average_response_time_by_service", json_data["summary"])
        self.assertIn("PROPHET", json_data["summary"]["average_response_time_by_service"])

if __name__ == "__main__":
    unittest.main()

