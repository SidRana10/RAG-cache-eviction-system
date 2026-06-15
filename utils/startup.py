"""handles startup and shutdown events for the application"""

from pinecone import Pinecone
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt
)
# custom modules
from .helpers import timer
from .config import pinecone_config

# configuration
pinecone_api_key = pinecone_config()['api key']
# database which holds all the answers
VECTOR_DB = None
CONFIGURED = False


@timer
@retry(wait=wait_random_exponential(10, 40), stop=stop_after_attempt(5))
async def make_connections():
    """initializes resources on startup"""
    global VECTOR_DB, MODEL, CONFIGURED
    # Initialize resources here
    # vector database
    if pinecone_api_key is None:
        raise ValueError(
            "Pinecone API key is not set in environment variables."
            )
    VECTOR_DB = Pinecone(api_key=pinecone_api_key, pool_threads=30)

    # status to inform us of successful connection on all parts
    CONFIGURED = True


@timer
def get_resources():
    """returns initialized resources"""
    # we want access to resource needed for the system to work,
    # so we check if they are initialized before returning
    if VECTOR_DB is None:
        raise ValueError(
            "Vector database connection has not been established."
            )
    return {"vector_db": VECTOR_DB}


def connection_success():
    """informs us of successful connection of all resources"""
    if CONFIGURED is False:
        raise ValueError(
            "resources not configured yet"
            )
    return CONFIGURED
