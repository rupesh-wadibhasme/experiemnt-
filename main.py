from flask import Flask, request, jsonify
from datetime import datetime
import backend.service_api as svc

app = Flask(__name__)

# ---- Route: Document Metrics ----
@app.route("/document-metrics", methods=["GET"])
def document_metrics_route():
    document_name = request.args.get("name")
    start = request.args.get("start")
    end = request.args.get("end")
    service_id = request.args.get("service_id")
    version = request.args.get("version")

    try:
        if document_name is not None:
            if document_name.strip() == "":
                return jsonify({"error": "Parameter 'name' cannot be empty."}), 400
            data = svc.get_document_metrics_by_name(document_name, service_id, version)

        elif start or end:
            if not (start and end):
                return jsonify({"error": "Both 'start' and 'end' parameters are required for date filtering."}), 400
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            data = svc.get_document_metrics_by_date(start_dt, end_dt, service_id, version)

        else:
            data = svc.get_all_document_metrics(service_id, version)

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Route: Chat Session Summary ----
@app.route("/chat-metrics", methods=["GET"])
def chat_metrics_route():
    session_id = request.args.get("session_id")
    service_id = request.args.get("service_id")
    version = request.args.get("version")

    try:
        if session_id:
            data = svc.get_chat_by_session_id(session_id, service_id, version)
        else:
            data = svc.get_all_chat_sessions(service_id, version)

        if not data:
            return jsonify({"message": "No chat data found."}), 404

        summary = svc.summarize_chat_sessions(data)
        return jsonify({"summary": summary, "data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Route: Feedback Update ----
@app.route("/feedback", methods=["POST"])
def update_feedback_api():
    """API endpoint to update feedback for a specific answer_id."""
    params = request.json
    session_id = params.get('session_id')
    answer_id = params.get('answer_id')
    feedback = params.get('feedback')

    if session_id is None or answer_id is None or feedback is None:
        return jsonify({"error": "Required parameters: session_id, answer_id, and feedback."}), 400

    try:
        svc.add_feedback(session_id, answer_id, feedback)
        return jsonify({
            'results': {
                'session_id': session_id,
                'answer_id': answer_id
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
