from datetime import datetime


def format_rest_response(output=None):
    timestamp = datetime.utcnow().isoformat()
    return {
        "role": "bot",
        "message": output,
        "timestamp": timestamp
    }
