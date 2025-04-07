import os
import uuid
from typing import List, Dict, Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

from helpers.logger import setup_logger

logger = setup_logger("app")

# Default configuration values
DEFAULTS = {
    "host": "localhost",
    "port": 6333,
    "collection": "source_texts",
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "openai_model": "text-embedding-ada-002",
    "vector_size": 384,
    "openai_vector_size": 1536,
    "distance": Distance.COSINE
}


class QdrantVectorStore:
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the Qdrant vector store using config dictionary.
        Supports both sentence-transformers and OpenAI embeddings.
        Lazy-loads heavy models to reduce app startup time.
        """
        config = config or {}
        qconf = config.get("vectordb", {}).get("qdrant", {})

        self.host = qconf.get("host", DEFAULTS["host"])
        self.port = qconf.get("port", DEFAULTS["port"])
        self.collection_name = qconf.get("collection_name", DEFAULTS["collection"])
        self.provider = qconf.get("provider", DEFAULTS["provider"]).lower()

        # Embedding model config
        if self.provider == "openai":
            self.embedding_model_name = qconf.get("embedding_model", DEFAULTS["openai_model"])
            self.vector_size = qconf.get("vector_size", DEFAULTS["openai_vector_size"])
        else:
            self.embedding_model_name = qconf.get("embedding_model", DEFAULTS["model"])
            self.vector_size = qconf.get("vector_size", DEFAULTS["vector_size"])

        self.distance = getattr(Distance, qconf.get("distance", DEFAULTS["distance"].name), DEFAULTS["distance"])

        self.embedding_model = None  # for sentence-transformers
        self.openai_client = None  # for OpenAI

        self.client = QdrantClient(host=self.host, port=self.port)

        logger.info(
            f"QdrantVectorStore initialized with provider='{self.provider}', "
            f"model='{self.embedding_model_name}', collection='{self.collection_name}', "
            f"vector_size={self.vector_size}, distance={self.distance.name}"
        )

    def _lazy_load_embedding_model(self):
        """
        Lazily loads the embedding model based on the provider.
        """
        if self.provider == "openai":
            if self.openai_client is None:
                logger.info("Loading OpenAI client for embeddings...")
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            if self.embedding_model is None:
                logger.info(f"Loading SentenceTransformer model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _encode(self, text: str) -> List[float]:
        """
        Encodes a text string into an embedding using the configured provider.
        """
        self._lazy_load_embedding_model()

        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model_name
            )
            return response.data[0].embedding
        else:
            return self.embedding_model.encode(text).tolist()

    def _create_collection_if_not_exists(self):
        """
        Ensures the Qdrant collection exists; creates it if it does not.
        """
        collections = self.client.get_collections()
        existing = [col.name for col in collections.collections]

        if self.collection_name not in existing:
            logger.info(f"Creating new Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
            )
        else:
            logger.debug(f"Qdrant collection '{self.collection_name}' already exists.")

    def insert_chunks(self, document_id: str, chunks: List[str]) -> int:
        """
        Encodes and inserts text chunks into the Qdrant collection.

        Returns:
            int: Number of chunks successfully inserted.
        """
        points = []

        for chunk in chunks:
            embedding = self._encode(chunk)
            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "text": chunk
                    }
                )
            )

        if points:
            self._create_collection_if_not_exists()
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Inserted {len(points)} chunk(s) into Qdrant collection '{self.collection_name}'.")
        else:
            logger.warning("No chunks to insert into Qdrant.")

        return len(points)

    def search_similar(self, text: str, threshold: float, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for chunks similar to the input text in Qdrant.

        Args:
            text (str): The query text.
            threshold (float): Minimum similarity score.
            limit (int): Maximum number of results.

        Returns:
            List[Dict[str, Any]]: List of matching texts and their scores.
        """
        query_vector = self._encode(text)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        matches = [
            {
                "text": result.payload.get("text", ""),
                "score": result.score
            }
            for result in results if result.score >= threshold
        ]

        logger.info(
            f"Search completed for query '{text[:30]}...'. Found {len(matches)} matches above threshold {threshold}.")
        return matches
