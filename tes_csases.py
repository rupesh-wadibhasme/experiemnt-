import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from service_chatbot import add_record_chatbot_service_defination, delete_chatbot_service_defination_by_service_id, \
update_chatbot_service_defination_records, view_records_by_service_id, get_audit_trail_by_service_id_version, \
clean_record, close_chatbot_service_defination_by_service_id, make_record_inactive, \
view_all_records_chatbot_service_defination



class TestAddRecordChatbotServiceDefinition(unittest.TestCase):


    @patch('service_chatbot.make_record_inactive')
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.view_records_by_service_id', return_value=[])
    @patch('service_chatbot.check_record_status', return_value=False)  # Closed record exists
    def test_add_when_closed_record_exists(self, mock_check_status, mock_view_records, mock_cosmos, mock_make_inactive):
        mock_add = MagicMock()
        mock_cosmos.return_value.add = mock_add

        add_record_chatbot_service_defination(
            session_id='123', service_id='svc1', version='v1', user='user1',
            blob_container='container', input_folder='folder',
            confluence_space='space', confluence_url='url',
            confluence_token='token'
        )

        mock_add.assert_called_once()
        mock_make_inactive.assert_called_once_with('svc1', 'v1', 'closed')

    @patch('service_chatbot.make_record_inactive')
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.view_records_by_service_id', return_value=[])
    @patch('service_chatbot.check_record_status', return_value=True)  # No closed record
    def test_successful_addition_no_closed_record(self, mock_check_status, mock_view_records, mock_cosmos, mock_make_inactive):
        mock_add = MagicMock()
        mock_cosmos.return_value.add = mock_add
        add_record_chatbot_service_defination(
            session_id='456', service_id='svc2', version='v2', user='user2',
            blob_container='container2', input_folder='folder2',
            confluence_space='space2', confluence_url='url2',
            confluence_token='token2'
        )

        mock_add.assert_called_once()
        mock_make_inactive.assert_called_once_with('svc2', 'v2', 'closed')

    @patch('service_chatbot.make_record_inactive')
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.view_records_by_service_id', return_value=[])
    @patch('service_chatbot.check_record_status', return_value=True)
    def test_addition_failure_triggers_inactive(self, mock_check_status, mock_view_records, mock_cosmos, mock_make_inactive):
        mock_cosmos.return_value.add.side_effect = Exception("DB error")

        with self.assertRaises(RuntimeError) as context:
            add_record_chatbot_service_defination(
                session_id='789', service_id='svc3', version='v3', user='user3',
                blob_container='container3', input_folder='folder3',
                confluence_space='space3', confluence_url='url3',
                confluence_token='token3'
            )
        self.assertIn("Error While Adding New Service Definition", str(context.exception))
        mock_make_inactive.assert_called_once_with('svc3', 'v3', 'closed')


    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_delete_successful(self, mock_client, mock_logger):
        # Setup mocks
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        # Mock query result
        mock_container.query_items.return_value = [
            {'id': 'session123', 'service_id': 'svc001', 'version': 'v1'}
        ]

        # Call function
        delete_chatbot_service_defination_by_service_id('svc001', 'v1')

        # Assertions
        mock_container.delete_item.assert_called_once_with(item='session123', partition_key='svc001')
        mock_logger.info.assert_called_with(
            "Records Permanently  DELETED From The Container. service_id:svc001, version:v1"
        )

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_delete_no_records_found(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        # No items returned
        mock_container.query_items.return_value = []

        with self.assertRaises(RuntimeError) as context:
            delete_chatbot_service_defination_by_service_id('svc002', 'v2')

        self.assertIn("Error while Hard Deleting a record in Service_id : svc002, version : v2", str(context.exception))
        mock_logger.info.assert_called_with(
            "Error while Hard Deleting a record in Service_id : svc002, version : v2"
        )

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_delete_exception_during_deletion(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.side_effect = Exception("DB error")

        with self.assertRaises(RuntimeError) as context:
            delete_chatbot_service_defination_by_service_id('svc003', 'v3')

        self.assertIn("Error while Hard Deleting a record in Service_id : svc003, version : v3", str(context.exception))
        mock_logger.info.assert_called_with(
            "Error while Hard Deleting a record in Service_id : svc003, version : v3"
        )

        
        
    @patch('service_chatbot.view_records_by_service_id', return_value=[])
    def test_no_existing_record(self, mock_view_records):
        with self.assertRaises(RuntimeError) as context:
            update_chatbot_service_defination_records(
                service_id='svc1', version='v1', user='user1',
                blob_container='container', input_folder='folder',
                confluence_space='space', confluence_url='url',
                confluence_token='token'
            )
        self.assertIn("No record found for service ID 'svc1' with version 'v1'", str(context.exception))

    @patch('service_chatbot.view_records_by_service_id', return_value=[{'service_id': 'svc1', 'version': 'v1'}])
    @patch('service_chatbot.check_record_status', return_value=False)
    def test_closed_record(self, mock_check_status, mock_view_records):
        with self.assertRaises(RuntimeError) as context:
            update_chatbot_service_defination_records(
                service_id='svc1', version='v1', user='user1',
                blob_container='container', input_folder='folder',
                confluence_space='space', confluence_url='url',
                confluence_token='token'
            )
        self.assertIn("This Service Definition is Closed", str(context.exception))

    @patch('service_chatbot.view_records_by_service_id', return_value=[{'service_id': 'svc1', 'version': 'v1'}])
    @patch('service_chatbot.check_record_status', return_value=True)
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.make_record_inactive')
    def test_successful_update(self, mock_make_inactive, mock_cosmos, mock_check_status, mock_view_records):
        mock_add = MagicMock()
        mock_cosmos.return_value.add = mock_add

        update_chatbot_service_defination_records(
            service_id='svc1', version='v1', user='user1',
            blob_container='container', input_folder='folder',
            confluence_space='space', confluence_url='url',
            confluence_token='token'
        )

        mock_add.assert_called_once()
        mock_make_inactive.assert_called_once_with('svc1', 'v1', 'active')
        args, kwargs = mock_add.call_args
        self.assertEqual(args[0]['service_id'], 'svc1')
        self.assertEqual(args[0]['version'], 'v1')
        self.assertEqual(args[0]['status'], 'active')

    @patch('service_chatbot.view_records_by_service_id', return_value=[{'service_id': 'svc1', 'version': 'v1'}])
    @patch('service_chatbot.check_record_status', return_value=True)
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.make_record_inactive')
    def test_update_failure(self, mock_make_inactive, mock_cosmos, mock_check_status, mock_view_records):
        mock_cosmos.return_value.add.side_effect = Exception("DB error")

        with self.assertRaises(RuntimeError) as context:
            update_chatbot_service_defination_records(
                service_id='svc1', version='v1', user='user1',
                blob_container='container', input_folder='folder',
                confluence_space='space', confluence_url='url',
                confluence_token='token'
            )

        self.assertIn("Error While Adding New Updated Record in Service Definition", str(context.exception))
        mock_make_inactive.assert_called_once_with('svc1', 'v1', 'active')


    
    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_active_records_with_version(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = [
            {'service_id': 'svc1', 'version': 'v1', 'status': 'active'}
        ]

        result = view_records_by_service_id('svc1', flag="True", version='v1')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'active')
        mock_logger.info.assert_called_with("Records fetched for service_id:svc1 and version v1.")

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_active_records_without_version(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = [
            {'service_id': 'svc1', 'status': 'active'}
        ]

        result = view_records_by_service_id('svc1', flag="True")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'active')
        mock_logger.info.assert_called_with("Records fetched for service_id:svc1.")

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_inactive_or_closed_records(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = [
            {'service_id': 'svc1', 'status': 'closed'}
        ]

        result = view_records_by_service_id('svc1', flag="False")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'closed')
        mock_logger.info.assert_called_with("Records fetched for service_id:svc1.")

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_no_records_found_with_version(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = []

        result = view_records_by_service_id('svc2', flag="True", version='v2')

        self.assertEqual(result, [])
        mock_logger.info.assert_called_with("No active records found for service ID 'svc2' and version 'v2'.")

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_no_records_found_without_version(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = []

        result = view_records_by_service_id('svc3', flag="False")

        self.assertEqual(result, [])
        mock_logger.info.assert_called_with("No inactive/closed records found for service ID 'svc3'.")


    def test_clean_record_removes_unwanted_keys(self):
        record = {
            "storage": {
                "blob_container": "container",
                "input_folder": "folder",
                "extra_key": "should_be_removed"
            }
        }
        required_keys = {"blob_container", "input_folder"}
        cleaned = clean_record(record, required_keys)
        self.assertIn("blob_container", cleaned["storage"])
        self.assertIn("input_folder", cleaned["storage"])
        self.assertNotIn("extra_key", cleaned["storage"])

    def test_clean_record_no_storage_key(self):
        record = {"id": "123"}
        required_keys = {"blob_container", "input_folder"}
        cleaned = clean_record(record, required_keys)
        self.assertEqual(cleaned, record)

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_get_audit_trail_success(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = [
            {
                "service_id": "svc1",
                "version": "v1",
                "time_stamp": "2025-08-01 10:00:00",
                "storage": {
                    "blob_container": "container",
                    "input_folder": "folder",
                    "extra": "remove"
                }
            },
            {
                "service_id": "svc1",
                "version": "v1",
                "time_stamp": "2025-08-01 09:00:00",
                "storage": {
                    "blob_container": "container2",
                    "input_folder": "folder2"
                }
            }
        ]

        result = get_audit_trail_by_service_id_version("svc1", "v1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["time_stamp"], "2025-08-01 09:00:00")
        self.assertNotIn("extra", result[1]["storage"])
        mock_logger.info.assert_called_with("Audit Trail Fetched for service_id:svc1, version:v1")

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_get_audit_trail_no_records(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        mock_container.query_items.return_value = []

        result = get_audit_trail_by_service_id_version("svc2", "v2")
        self.assertEqual(result, [])

    
    @patch('service_chatbot.make_record_inactive')
    @patch('service_chatbot.logger')
    @patch('service_chatbot.CosmosDBTable')
    @patch('service_chatbot.check_record_status', return_value=True)
    @patch('service_chatbot.client')
    def test_soft_delete_success(self, mock_client, mock_check_status, mock_cosmos, mock_logger, mock_make_inactive):
        mock_container = MagicMock()
        mock_items = [{
            'id': 'session123',
            'status': 'active',
            'storage': {
                'blob_container': 'container',
                'input_folder': 'folder',
                'confluence_space': 'space',
                'confluence_url': 'url',
                'confluence_token': 'token'
            },
            'additional_versions': [],
            'user': 'user1'
        }]
        mock_container.query_items.return_value = mock_items
        mock_client.get_database_client.return_value.get_container_client.return_value = mock_container

        close_chatbot_service_defination_by_service_id('svc1', 'v1')
        mock_cosmos.return_value.add.assert_called_once()
        mock_make_inactive.assert_called_once_with('svc1', 'v1', 'closed')

    @patch('service_chatbot.check_record_status', return_value=False)
    def test_soft_delete_already_closed(self, mock_check_status):
        with self.assertRaises(RuntimeError) as context:
            close_chatbot_service_defination_by_service_id('svc1', 'v1')
        self.assertIn("This Service Definition is Closed", str(context.exception))

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_make_record_inactive_success(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_items = [
            {'id': '1', 'time_stamp': '2023-01-01 10:00:00', 'status': 'active'},
            {'id': '2', 'time_stamp': '2023-01-02 10:00:00', 'status': 'active'}
        ]
        mock_container.query_items.return_value = mock_items
        mock_client.get_database_client.return_value.get_container_client.return_value = mock_container
        make_record_inactive('svc1', 'v1')
        mock_container.replace_item.assert_called_once_with(item='1', body=mock_items[0])
        self.assertEqual(mock_items[0]['status'], 'inactive')

    @patch('service_chatbot.logger')
    @patch('service_chatbot.client')
    def test_make_record_inactive_error(self, mock_client, mock_logger):
        mock_container = MagicMock()
        mock_container.query_items.side_effect = Exception("DB error")
        mock_client.get_database_client.return_value.get_container_client.return_value = mock_container

        make_record_inactive('svc1', 'v1')
        mock_logger.error.assert_called_with("Error while updating the record to inactive status in svc1, v1")


    @patch('service_chatbot.CosmosDBTable')
    def test_view_all_records_success(self, mock_cosmos_table):
        mock_cosmos_table.return_value = [
            {'service_id': 'svc1', 'version': 'v1'},
            {'service_id': 'svc2', 'version': 'v2'}
        ]

        result = view_all_records_chatbot_service_defination()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['service_id'], 'svc1')
        self.assertEqual(result[1]['version'], 'v2')

    @patch('service_chatbot.CosmosDBTable')
    def test_view_all_records_empty(self, mock_cosmos_table):
        mock_cosmos_table.return_value = []

        result = view_all_records_chatbot_service_defination()
        self.assertEqual(result, [])



if __name__ == '__main__':
    unittest.main()
