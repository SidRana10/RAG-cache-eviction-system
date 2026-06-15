"""Logic for retrieving relevant docs"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from ..utils.startup import get_resources
from ..utils.logger import logger
from ..utils.helpers import timer, returns
from ..utils.config import huggingface_config


# load resources needed for retrieval
resources = get_resources()

# vector_db is needed for retrieval,
# so we make it available at the module level
vector_db = resources["vector_db"]

# Initialize embedding model for query embedding
_embedding_config = huggingface_config()
_embedding_model = HuggingFaceEmbeddings(
    model_name=_embedding_config.get('embedding model',
                                      'sentence-transformers/all-MiniLM-L6-v2')
)


# retrieves context to make the response knowledgable
@timer  # to keep track of time taken for retrieval
@returns(list)  # we expect a list of relevant documents to be returned
def retrieve_context(query: str, namespace: str = 'None', top_k: int = 5):
    """retrieves relevant context for a given query"""
    # perform retrieval using the vector database
    if namespace == 'None':
        logger.warning(
            "No namespace provided for retrieval. Using default namespace."
        )
        raise ValueError("Namespace is required for retrieval.")

    # Step 1: Embed the query string
    try:
        query_embedding = _embedding_model.embed_query(query)
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        raise

    # Step 2: Query the vector store with the embedding
    try:
        results = vector_db.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace)
        return results
    except Exception as e:
        logger.error("Vector DB query failed: %s", e)
        raise
