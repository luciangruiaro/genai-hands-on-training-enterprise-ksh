# Qdrant Vector Store Integration

This module integrates [Qdrant](https://qdrant.tech/) as a vector database for storing and retrieving semantic text embeddings. It uses the `sentence-transformers` library for embedding generation and the `qdrant-client` for communicating with Qdrant.

---

## 🧠 What This Module Does

- Connects to a Qdrant server (Docker or remote)
- Automatically creates a collection if it doesn't exist
- Uses sentence-transformer models to encode text chunks into dense vectors
- Performs similarity search against Qdrant and returns top matching texts

---

## ⚙️ Configuration Example (`config.yaml`)

```yaml
vectordb:
  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "source_texts"
    embedding_model: "all-MiniLM-L6-v2"  # Options below
    vector_size: 384                     # Must match model's output
    distance: "COSINE"                   # Options: COSINE, EUCLID, DOT
```

---

## 🔍 Recommended Embedding Models

Below are commonly used models you can configure under `embedding_model:`.

| Model | Description | Output Dim |
|-------|-------------|------------|
| `all-MiniLM-L6-v2` | ⚡ Fast + lightweight | `384` |
| `all-MiniLM-L12-v2` | Slightly better, slower | `384` |
| `paraphrase-MiniLM-L6-v2` | Paraphrase detection | `384` |
| `paraphrase-mpnet-base-v2` | High quality, nuanced | `768` |
| `multi-qa-MiniLM-L6-cos-v1` | Optimized for QA | `384` |
| `multi-qa-mpnet-base-dot-v1` | QA + dot product support | `768` |
| `distiluse-base-multilingual-cased-v1` | 🌍 Multilingual (15+ langs) | `512` |
| `LaBSE` | Google multilingual | `768` |
| `intfloat/e5-small-v2` | 🧪 Newer generation, small | `384` |
| `intfloat/e5-base-v2` | Newer + better than MiniLM | `768` |

> ⚠️ Make sure to match `vector_size` with the correct output dimensions.

---

## 💡 Notes

- Some models (like `e5-*`) expect **prompt formatting**, e.g.:
  ```
  "query: What is a language model?"
  "passage: A language model predicts the next word..."
  ```
  You may need to wrap `.encode()` calls to add these prefixes.

- The default distance metric is `COSINE`. You can also use `EUCLID` or `DOT` if your model was trained accordingly.

---

## 🛠 Usage Example

```python
from integrations.vectordb.qdrant.qdrant_vectorstore import QdrantVectorStore

qdrant = QdrantVectorStore(config)

qdrant.insert_chunks("doc-123", ["First paragraph", "Second paragraph"])

results = qdrant.search_similar("What is a paragraph?", threshold=0.8)
```

---

## 📚 References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
