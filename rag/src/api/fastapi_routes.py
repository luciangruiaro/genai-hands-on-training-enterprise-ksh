import uuid

from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas.chat_schema import ChatRequest
from helpers.logger import setup_logger
from helpers.utils import format_rest_response
from integrations.vectordb.qdrant.qdrant_vectorstore import QdrantVectorStore
from service.agent_ai import AgentAI
from service.hello_service import HelloService
from service.text_chunking import TextChunkingService

logger = setup_logger("app")


def create_fastapi_app(config):
    app = FastAPI()
    app.state.config = config
    cors_setup(app)

    hello_service = HelloService(config)
    templates = Jinja2Templates(directory='templates')
    app.mount("/static", StaticFiles(directory="static"), name="static")

    agent = AgentAI(config)
    chunker = TextChunkingService(config)
    vector_store = QdrantVectorStore(config)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            user_input = request.message.strip()
            if not user_input:
                raise HTTPException(status_code=400, detail="Empty message")
            return JSONResponse(content=format_rest_response(agent.respond(user_input)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/history")
    async def get_history():
        try:
            return JSONResponse(content={"history": agent.get_memories()})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chunk")
    async def chunk_text(payload: dict = Body(...)):
        logger.info("Received request: chunk-text with raw text.")
        try:
            text = payload.get("text", "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="Missing 'text' in request body")

            chunks = chunker.chunk_text(text)
            return JSONResponse(content={"chunks": chunks})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/add-to-qdrant")
    async def add_to_qdrant(payload: dict = Body(...)):
        logger.info("Received request to add data to Qdrant.")
        try:
            text = payload.get("text", "").strip()
            document_id = payload.get("document_id", "").strip() or str(uuid.uuid4())

            if not text:
                raise HTTPException(status_code=400, detail="Missing 'text' in request body")

            chunks = chunker.chunk_text(text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks generated from input text.")

            inserted_count = vector_store.insert_chunks(document_id, chunks)

            return JSONResponse(content={
                "document_id": document_id,
                "chunks_added": inserted_count,
                "chunk_preview": chunks[:3]  # Optional: preview first 3 chunks
            })
        except Exception as e:
            logger.exception("Error in /add-to-qdrant route")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search-qdrant")
    async def search_qdrant(payload: dict = Body(...)):
        logger.info("Received request to search in Qdrant.")
        try:
            text = payload.get("text", "").strip()
            threshold = float(payload.get("threshold", 0.75))
            limit = int(payload.get("limit", 5))

            if not text:
                raise HTTPException(status_code=400, detail="Missing 'text' in request body")

            results = vector_store.search_similar(text, threshold=threshold, limit=limit)

            return JSONResponse(content={"results": results})
        except Exception as e:
            logger.exception("Error in /search-qdrant route")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def cors_setup(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
