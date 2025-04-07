import json

from config.knowledge_manager import knowledge_curr
from helpers.logger import setup_logger
from integrations.llm.llm_interface import LLMClient
from integrations.vectordb.qdrant.qdrant_vectorstore import QdrantVectorStore

USER = "user"
BOT = "bot"


class AgentAI:
    def __init__(self, config: dict):
        self.logger = setup_logger("app")
        self.config = config
        self.constants = config.get("constants", {})
        self.llm = LLMClient(config)

        self.user_role = self.constants.get("user", USER)
        self.bot_role = self.constants.get("bot", BOT)

    def respond(self, user_input: str) -> str:
        self.logger.info(f"Agent received input: {user_input}")
        return self._handle_conversation(user_input)

    def _handle_conversation(self, user_input: str) -> str:
        knowledge = self._knowledge_setup(user_input)
        context_window = self.llm.context_window
        max_tokens = self.llm.max_tokens

        # remaining_tokens = calculate_remaining_tokens(
        #     user_input=user_input,
        #     knowledge=knowledge,
        #     context_window=context_window,
        #     max_tokens=max_tokens,
        #     user_role=self.user_role
        # )

        initial_response = self.llm.ask(
            user_input=user_input,
            knowledge=knowledge
        )

        return initial_response

    def _knowledge_setup(self, user_input: str) -> str:
        nl = "\n"
        base = "You are an AI Agent, called RAG_Chatbot. Give your best to respond given your knowledge."

        source = self.config.get("knowledge", {}).get("source", "file")
        threshold = self.config.get("knowledge", {}).get("threshold", 0.7)
        limit = self.config.get("knowledge", {}).get("limit", 3)

        if source == "qdrant":
            try:
                if not hasattr(self, "vector_store"):
                    self.vector_store = QdrantVectorStore(self.config)
                results = self.vector_store.search_similar(user_input, threshold=threshold, limit=limit)
                if results:
                    chunks = "\n".join([f"- {r['text']}" for r in results])
                    return f"{base}\nThis is what you know from database:\n{chunks}"
                else:
                    self.logger.warning("Qdrant returned no results. Falling back to file knowledge.")
            except Exception as e:
                self.logger.error(f"Failed to fetch knowledge from Qdrant: {e}")
                # fall back

        # Fallback to JSON knowledge
        knowledge = json.dumps(knowledge_curr.get_knowledge(), indent=2)
        return f"{base}\nThis is what you know:\n{knowledge}"
