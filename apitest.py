# test_monitoring_api.py

import unittest
from unittest.mock import patch, MagicMock
import monitoring_api  # Imports your module

class MonitoringAPITestCase(unittest.TestCase):

    @patch("monitoring_api.load_config")
    @patch("monitoring_api.CosmosClient")
    def test_get_all_documents(self, mock_cosmos_client, mock_load_config):
        # Mock config
        mock_load_config.return_value = {
            "DB_URL": "mock-url",
            "DB_KEY": "mock-key",
            "DB_NAME": "mock-db"
        }

        # Mock Cosmos container
        mock_container = MagicMock()
        mock_container.read_all_items.return_value = [{"id": "doc1"}]

        # Patch Cosmos client behavior
        mock_db_client = MagicMock()
        mock_db_client.get_container_client.return_value = mock_container
        mock_cosmos_client.return_value.get_database_client.return_value = mock_db_client

        # Re-run config and client setup
        monitoring_api.CONFIG = mock_load_config.return_value
        monitoring_api.client = mock_cosmos_client.return_value

        result = monitoring_api.get_all_documents()
        self.assertEqual(result, [{"id": "doc1"}])

    @patch("monitoring_api.load_config")
    @patch("monitoring_api.CosmosClient")
    def test_summarize_chat_sessions(self, mock_cosmos_client, mock_load_config):
        mock_load_config.return_value = {
            "DB_URL": "mock-url",
            "DB_KEY": "mock-key",
            "DB_NAME": "mock-db"
        }

        # Mock chat items
        items = [
            {"service_id": "PROPHET", "session_id": "s1", "response_time": 5.0, "feedback": "good"},
            {"service_id": "PROPHET", "session_id": "s1", "response_time": 7.0, "feedback": "neutral"},
            {"service_id": "PROPHET", "session_id": "s2", "response_time": 6.0, "feedback": "bad"},
        ]

        summary = monitoring_api.summarize_chat_sessions(items)

        self.assertIn("average_response_time_by_service", summary)
        self.assertEqual(summary["average_response_time_by_service"].get("PROPHET"), 6.0)
        self.assertIn("feedback_distribution_by_service", summary)
        self.assertEqual(summary["feedback_distribution_by_service"]["PROPHET"]["good"], 1)
        self.assertEqual(summary["average_queries_per_session"], 1.5)

if __name__ == '__main__':
    unittest.main()
