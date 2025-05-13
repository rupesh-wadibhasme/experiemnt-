from flask import Flask, request, jsonify
from azure.cosmos import CosmosClient
from datetime import datetime
import os

# Configuration 

COSMOS_CONNECTION_STRING = "AccountEndpoint=https://<your-account>.documents.azure.com:443/;AccountKey=<your-key>"
COSMOS_DATABASE_NAME = "KnowledgeBase"
COSMOS_CONTAINER_NAME = "Document_Metrics"

class AnalyticsAPI:
    def __init__(self, conn_str, db_name, container_name):
        self.client = CosmosClient.from_connection_string(conn_str)
        self.database = self.client.get_database_client(db_name)
        self.container = self.database.get_container_client(container_name)

    def get_all_items(self):
        return list(self.container.read_all_items())

    def get_item_by_id(self, item_id):
        query = f"SELECT * FROM c WHERE c.id = '{item_id}'"
        return list(self.container.query_items(query=query, enable_cross_partition_query=True))

    def get_items_by_date_range(self, start: datetime, end: datetime):
        query = f"""
        SELECT * FROM c
        WHERE c.timestamp >= '{start.isoformat()}' AND c.timestamp <= '{end.isoformat()}'
        """
        return list(self.container.query_items(query=query, enable_cross_partition_query=True))

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
analytics = AnalyticsAPI(COSMOS_CONNECTION_STRING, COSMOS_DATABASE_NAME, COSMOS_CONTAINER_NAME)

@app.route("/Doc_Metrix", methods=["GET"])
def analytics_route():
    item_id = request.args.get("id")
    start = request.args.get("start")
    end = request.args.get("end")

    try:
        if item_id:
            return jsonify(analytics.get_item_by_id(item_id))
        elif start and end:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            return jsonify(analytics.get_items_by_date_range(start_dt, end_dt))
        else:
            return jsonify(analytics.get_all_items())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
