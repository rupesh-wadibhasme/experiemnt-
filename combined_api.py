from flask import Flask, request, jsonify
from azure.cosmos import CosmosClient
from datetime import datetime
from collections import defaultdict, Counter
from backend.common import load_config  # Assuming your config function is here

# ---- Load Secure Config ----
CONFIG = load_config()
DB_NAME = CONFIG['DB_NAME']
DOCUMENT_METRICS_CONTAINER = "Document_Metrics"
CHAT_METRICS_CONTAINER = "Chat_Session_Metrics"

app = Flask(__name__)

# ---- Document Analytics Class ----
class DocumentAnalytics:
    def __init__(self, client):
        self.container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)

    def get_all_items(self):
        return list(self.container.read_all_items())

    def get_item_by_id(self, item_id):
        query = f"SELECT * FROM c WHERE c.id = '{item_id}'"
        return list(self.container.query_items(query=query, enable_cross_partition_query=True))

    def get_items_by_date_range(self, start, end):
        query = f"""
        SELECT * FROM c
        WHERE c.timestamp >= '{start.isoformat()}' AND c.timestamp <= '{end.isoformat()}'
        """
        return list(self.container.query_items(query=query, enable_cross_partition_query=True))

# ---- Chat Session Analytics Class ----
class ChatAnalytics:
    def __init__(self, client):
        self.container = client.get_database_client(DB_NAME).get_container_client(CHAT_METRICS_CONTAINER)

    def get_all_items(self):
        return list(self.container.read_all_items())

    def get_item_by_id(self, item_id):
        query = f"SELECT * FROM c WHERE c.id = '{item_id}'"
        return list(self.container.query_items(query=query, enable_cross_partition_query=True))

    def summarize(self, items):
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

# ---- Cosmos Setup ----
client = CosmosClient(CONFIG['DB_URL'], credential=CONFIG['DB_KEY'])
doc_api = DocumentAnalytics(client)
chat_api = ChatAnalytics(client)

# ---- Route: Document Analytics ----
@app.route("/analytics", methods=["GET"])
def analytics_route():
    item_id = request.args.get("id")
    start = request.args.get("start")
    end = request.args.get("end")
    try:
        if item_id:
            data = doc_api.get_item_by_id(item_id)
        elif start and end:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            data = doc_api.get_items_by_date_range(start_dt, end_dt)
        else:
            data = doc_api.get_all_items()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Route: Chat Session Summary ----
@app.route("/chat-metrics", methods=["GET"])
def chat_metrics_route():
    item_id = request.args.get("id")
    try:
        if item_id:
            data = chat_api.get_item_by_id(item_id)
        else:
            data = chat_api.get_all_items()
        summary = chat_api.summarize(data)
        return jsonify({"summary": summary, "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- App Launcher ----
if __name__ == "__main__":
    app.run(debug=True)
