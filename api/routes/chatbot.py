"""Defines API endpoints"""
import asyncio
from fastapi import HTTPException, APIRouter, BackgroundTasks
from pathlib import Path

# custom modules
from ...core.utils.startup import connection_success
from ..middleware.schema import UserRequest, ChatResponse
from ...core.generation.rag_pipeline import RAGPipeline
from ...core.utils.helpers import timer, timeout, export_to_csv
from ...core.utils.logger import logger
from ...core.utils.config import benchmark_const


# configurations
constants = benchmark_const()
save_path = constants['path']
filename = constants['filename']

chatbot = APIRouter()


@timeout(30)
@chatbot.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
        user_request: UserRequest, background_tasks: BackgroundTasks):
    """endpoint to facilitate conversations"""
    try:
        # we create an instance of the RAGPipeline
        # then run the pipeline to generate an answer
        chat = RAGPipeline()
        response = await asyncio.to_thread(
            chat.generate_answer, user_request.prompt)

        # export interaction for evaluation
        async def store_interaction():
            data = [
                    user_request.prompt,
                    response['answer'],
                    response['sources'][0]['text']]
            export_to_csv(data, save_path, filename)
            logger.info("background task completed: data exported")

        # activated only when running locally
        if (Path.home() / "scripts").exists():
            background_tasks.add_task(store_interaction)

        logger.info("results returned successfully")
        await asyncio.sleep(1)  # Simulate delay
        return ChatResponse(
            response=response['answer'], sources=response['sources'])
    except Exception as exc:
        logger.error("unsuccessful", extra={"error": type(exc).__name__})
        raise HTTPException(
            status_code=500, detail="could not provide a response") from exc


@timer
@chatbot.get("/services/health")
async def service_health():
    """gets initialization status of core resources"""
    try:
        # we wait for the connection to be successful
        # this is a blocking operation
        # so we use asyncio.wait_for to timeout after 10 seconds
        status = await asyncio.wait_for(
            connection_success(), timeout=10)
        if status is False:
            logger.warning(
                "Health check failed: external services not initialized")
            raise HTTPException(status_code=500, detail="Service not ready")

        return {
            "status": "healthy",
            "service": "running",
            "services_initialized": status
        }
    except asyncio.TimeoutError as exc:
        logger.error("Health check failed: timeout while checking services")
        raise HTTPException(
            status_code=408, detail="Service health check timed out") from exc

    except Exception as exc:
        logger.error(
            "Health check failed", extra={"error": type(exc).__name__})
        raise HTTPException(
            status_code=500, detail="Service health check failed") from exc
