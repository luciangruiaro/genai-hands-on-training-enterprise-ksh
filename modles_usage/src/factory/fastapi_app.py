from fastapi import FastAPI, Request
import uvicorn

from service.responder import generate_response

app = FastAPI()


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "")
    return {"response": generate_response(message)}


def start_fastapi():
    uvicorn.run("factory.fastapi_app:app", host="127.0.0.1", port=5000, reload=True)
