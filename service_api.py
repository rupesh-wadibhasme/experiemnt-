from datetime import datetime
from collections import defaultdict, Counter
from azure.cosmos import CosmosClient
from backend.common import load_config  # Uses secure config

# ---- Load Config ----
CONFIG = load_config()
DB_NAME = CONFIG['DB_NAME']
DOCUMENT_METRICS_CONTAINER = "Document_Metrics"
CHAT_METRICS_CONTAINER = "Chat_Session_Metrics"

# ---- Cosmos Setup ----
client = CosmosClient(CONFIG['DB_URL'], credential=CONFIG['DB_KEY'])

# ---- Document Analytics ----
def get_all_document_metrics(service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = "SELECT * FROM c"
    if service_id and version:
        query += f" WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def get_document_metrics_by_name(document_name, service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.url LIKE '%{document_name}%'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def get_document_metrics_by_date(start, end, service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.timestamp >= '{start.isoformat()}' AND c.timestamp <= '{end.isoformat()}'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

# ---- Chat Analytics ----
def get_all_chat_sessions(service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(CHAT_METRICS_CONTAINER)
    query = "SELECT * FROM c"
    if service_id and version:
        query += f" WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def get_chat_by_session_id(session_id, service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(CHAT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.session_id = '{session_id}'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def summarize_chat_sessions(items):
    service_response_times = defaultdict(list)
    feedback_counts = defaultdict(lambda: Counter({'good': 0, 'bad': 0, 'neutral': 0, 'none': 0}))
    session_query_count = Counter()

    for item in items:
        service = item.get("service_id", "unknown")
        session = item.get("session_id", "unknown")
        session_query_count[session] += 1

        rt = item.get("response_time")
        if isinstance(rt, (float, int)):
            service_response_times[service].append(rt)

        feedback = item.get("feedback", "").lower()
        if feedback not in ['good', 'bad', 'neutral']:
            feedback = 'none'
        feedback_counts[service][feedback] += 1

    avg_rt = {s: round(sum(times)/len(times), 2) for s, times in service_response_times.items()}
    avg_queries = round(sum(session_query_count.values()) / max(len(session_query_count), 1), 2)

    return {
        "average_response_time_by_service": avg_rt,
        "feedback_distribution_by_service": feedback_counts,
        "average_queries_per_session": avg_queries
    }

def get_document_count_by_service_id():
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)

    # Query all service_id values
    query = "SELECT c.service_id FROM c"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    # Count how many times each service_id appears
    counts = defaultdict(int)
    for item in items:
        service_id = item.get("service_id", "unknown")
        counts[service_id] += 1

    return dict(counts)

