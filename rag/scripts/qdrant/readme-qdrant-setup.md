# Qdrant Setup

This folder contains scripts to run [Qdrant](https://qdrant.tech/), a high-performance vector database for semantic
search, recommendation engines, and large-scale LLM-based applications.

---

## 🚀 Run Locally with Docker Compose

To start Qdrant using Docker Compose:

```bash
bash start-qdrant.sh
```

This will:

- Pull the latest `qdrant/qdrant` image
- Run it in a container named `genai-handson-qdrant`
- Expose the following ports:
    - `6333`: REST API
    - `6334`: gRPC API

---

## 🔧 Manual Docker Command (No Compose Required)

If you prefer to run Qdrant without Docker Compose or YAML files:

Linux:

```bash
docker run -d \
  --name qdrant-standalone \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

Windows:

```bash
docker run -d --name qdrant-standalone -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

> 🔁 You can replace `latest` with a specific version like `v1.7.2` for better stability.

---

## 📦 Data Persistence

Qdrant stores vector collections in a Docker volume named `qdrant_data`.

---

## 📍 Default Endpoints

- **REST API**: [http://localhost:6333](http://localhost:6333)
- **gRPC API**: `localhost:6334`

---

## 🧪 Test It

To confirm Qdrant is running, you can check the `/health` endpoint:

```bash
curl http://localhost:6333/health
```

---

## 📚 More Info

- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Python Client](https://qdrant.tech/documentation/overview/)
