"""FastAPI entry point: RAG chat and health routes under /v1."""

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt)


from ..core.utils.startup import make_connections
from .routes.chatbot import chatbot

# making resources available upon startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """initializes resources on startup"""
    # Initialize resources here
    await make_connections()
    yield
    


# api client
@retry(wait=wait_random_exponential(10, 40), stop=stop_after_attempt(5))
def create_app():
    """creates the client application and includes the routes"""
    app = FastAPI(lifespan=lifespan)
    # include API endpoints
    app.include_router(chatbot, prefix="/v1")
    return app


# client application
server = create_app()

if __name__ == '__main__':
    uvicorn.run('api.server:server', host='127.0.0.1', port=8000, reload=True)
