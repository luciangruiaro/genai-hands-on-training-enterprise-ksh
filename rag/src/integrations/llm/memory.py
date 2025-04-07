from datetime import datetime, timezone
from collections import deque


class AgentMemory:
    def __init__(self, max_size: int = 100):
        """
        Initializes a conversation memory buffer.

        :param max_size: Maximum number of messages to store.
        """
        self._history = deque(maxlen=max_size)

    def add_message(self, role: str, content: str):
        """
        Adds a message to the memory history.

        :param role: Role of the sender (e.g., "user", "assistant").
        :param content: Text content of the message.
        """
        if not role or not content:
            raise ValueError("Both role and content must be provided.")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._history.append(message)

    def get_history(self, limit: int = None):
        """
        Retrieves the most recent messages.

        :param limit: Max number of messages to return.
        :return: List of message dictionaries.
        """
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("Limit must be a positive integer.")
            return list(self._history)[-limit:]
        return list(self._history)

    def clear(self):
        """Clears the conversation history."""
        self._history.clear()

    def to_dict(self):
        """
        Converts the memory to a list of dictionaries.

        :return: List of message dictionaries.
        """
        return list(self._history)

    def __repr__(self):
        return f"ConversationMemory(entries={len(self._history)})"
