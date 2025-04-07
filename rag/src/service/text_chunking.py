from typing import List, Optional

import stanza
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from helpers.chunk_exporter import export_chunks_to_json
from helpers.logger import setup_logger

logger = setup_logger("app")

# --- Constants / Defaults ---
DEFAULT_ENABLE_VARIABLE = True
DEFAULT_ENABLE_SEMANTIC = False
DEFAULT_MAX_SENTENCES = 5
DEFAULT_MAX_WORDS = 120
DEFAULT_MAX_CHARACTERS = 1000
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_THRESHOLD = 70  # in percent


class TextChunkingService:
    def __init__(self, config: dict):
        chunk_conf = config.get("chunking", {})

        self.enable_variable = chunk_conf.get("enable_variable", DEFAULT_ENABLE_VARIABLE)
        self.enable_semantic = chunk_conf.get("enable_semantic", DEFAULT_ENABLE_SEMANTIC)

        self.max_sentences = chunk_conf.get("max_sentences", DEFAULT_MAX_SENTENCES)
        self.max_words = chunk_conf.get("max_words", DEFAULT_MAX_WORDS)
        self.max_characters = chunk_conf.get("max_characters", DEFAULT_MAX_CHARACTERS)

        self.semantic_model_name = chunk_conf.get("semantic_embed_model", DEFAULT_EMBED_MODEL)
        self.semantic_threshold = chunk_conf.get("semantic_breakpoint_threshold", DEFAULT_SEMANTIC_THRESHOLD)

        self._semantic_model: Optional[HuggingFaceEmbedding] = None
        self._stanza_nlp: Optional[stanza.Pipeline] = None

    def chunk_text(self, text: str) -> List[str]:
        """
        Orchestrates both variable and semantic chunking based on config.
        Returns a de-duplicated list of text chunks.
        """
        logger.info("Starting text chunking...")
        chunks: List[str] = []

        if self.enable_variable:
            variable_chunks = self._variable_chunking(text)
            logger.debug(f"Variable chunking produced {len(variable_chunks)} chunk(s).")
            chunks.extend(variable_chunks)
        else:
            logger.debug("Variable chunking disabled.")

        if self.enable_semantic:
            semantic_chunks = self._semantic_chunking(text)
            logger.debug(f"Semantic chunking produced {len(semantic_chunks)} chunk(s).")
            chunks.extend(semantic_chunks)
        else:
            logger.debug("Semantic chunking disabled.")

        # Remove duplicates while preserving order
        unique_chunks = list(dict.fromkeys(chunks))
        export_path = export_chunks_to_json(text, unique_chunks, semantic_embed_model=self.semantic_model_name)

        logger.info(f"Chunking completed. Exported {len(unique_chunks)} chunk(s) to {export_path}.")

        if not unique_chunks:
            logger.warning("No chunks were generated from the input text.")

        return unique_chunks

    def _get_embedding_model(self):
        if "openai" in self.semantic_model_name.lower():
            model_id = self.semantic_model_name.replace("openai/", "", 1)
            logger.info(f"Using OpenAIEmbedding model: {model_id}")
            return OpenAIEmbedding(model=model_id)
        else:
            logger.info(f"Using HuggingFaceEmbedding model: {self.semantic_model_name}")
            return HuggingFaceEmbedding(model_name=self.semantic_model_name)

    def _variable_chunking(self, text: str) -> List[str]:
        """
        Sentence-based chunking using Stanza NLP.
        """
        logger.info("Performing variable chunking...")

        if not self._stanza_nlp:
            logger.info("Lazy-loading Stanza pipeline...")
            self._stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)

        doc = self._stanza_nlp(text)
        sentences = [s.text for s in doc.sentences]

        chunks, current_chunk = [], []
        word_count = char_count = 0

        for sentence in sentences:
            words = sentence.split()
            word_len = len(words)
            char_len = len(sentence)

            if (
                    len(current_chunk) + 1 > self.max_sentences or
                    word_count + word_len > self.max_words or
                    char_count + char_len > self.max_characters
            ):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                word_count = word_len
                char_count = char_len
            else:
                current_chunk.append(sentence)
                word_count += word_len
                char_count += char_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Variable chunking completed. Total chunks: {len(chunks)}")
        return chunks

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Embedding-based semantic chunking using a percentile threshold.
        """
        logger.info("Performing semantic chunking...")

        if not self._semantic_model:
            logger.info(f"Loading semantic embedding model: {self.semantic_model_name}")
            self._semantic_model = self._get_embedding_model()

        # Convert to percentile if given as float (e.g. 0.7 → 70)
        threshold = int(self.semantic_threshold * 100) if isinstance(self.semantic_threshold,
                                                                     float) else self.semantic_threshold

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=threshold,
            embed_model=self._semantic_model
        )

        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        chunks = [node.get_content() for node in nodes]

        logger.info(f"Semantic chunking completed with threshold {threshold}. Total chunks: {len(chunks)}")
        return chunks
