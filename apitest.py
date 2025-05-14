from unittest.mock import patch

@patch('backend.service_api.get_all_documents')
def test_analytics_mocked(mock_get, client):
    mock_get.return_value = [{"id": "doc1"}]
    response = client.get("/analytics")
    assert response.status_code == 200
    assert response.json == [{"id": "doc1"}]

@patch('backend.service_api.get_all_chat_sessions')
@patch('backend.service_api.summarize_chat_sessions')
def test_chat_metrics_mocked(mock_summary, mock_get, client):
    mock_get.return_value = [{"id": "chat1"}]
    mock_summary.return_value = {"average": 1}
    response = client.get("/chat-metrics")
    assert response.status_code == 200
    assert "summary" in response.json
    assert "data" in response.json
