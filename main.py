from flask import Flask, request, jsonify
from datetime import datetime
import backend.service_api as svc

app = Flask(__name__)

# ---- Route: Document Analytics ----
@app.route("/analytics", methods=["GET"])
def analytics_route():
    item_id = request.args.get("id")
    start = request.args.get("start")
    end = request.args.get("end")
    try:
        if item_id:
            data = svc.get_document_by_id(item_id)
        elif start and end:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            data = svc.get_documents_by_date(start_dt, end_dt)
        else:
            data = svc.get_all_documents()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Route: Chat Session Summary ----
@app.route("/chat-metrics", methods=["GET"])
def chat_metrics_route():
    item_id = request.args.get("id")
    try:
        if item_id:
            data = svc.get_chat_by_id(item_id)
        else:
            data = svc.get_all_chat_sessions()
        summary = svc.summarize_chat_sessions(data)
        return jsonify({"summary": summary, "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
