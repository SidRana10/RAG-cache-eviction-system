"""Vector store wrapper for Pinecone operations"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from ..utils.config import pinecone_config
from ..utils.logger import logger


class VectorStore:
    """Wrapper around Pinecone vector database operations."""

    def __init__(self, pinecone_client: Optional[Pinecone] = None):
        """Initialize with Pinecone client or create one."""
        if pinecone_client is None:
            config = pinecone_config()
            self.client = Pinecone(api_key=config['api key'], pool_threads=30)
        else:
            self.client = pinecone_client
        self.config = pinecone_config()
        self.index_name = self.config['index']
        self.namespace = self.config['namespace']

    def get_index(self):
        """Get the Pinecone index object."""
        return self.client.Index(self.index_name)

    def query(self, query_vector: List[float], top_k: int = 5, namespace: str = None, include_metadata: bool = True) -> Dict[str, Any]:
        """Query the vector store for similar vectors."""
        index = self.get_index()
        namespace = namespace or self.namespace
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = None) -> Dict[str, Any]:
        """Upsert vectors into the store."""
        index = self.get_index()
        namespace = namespace or self.namespace
        try:
            response = index.upsert(vectors=vectors, namespace=namespace)
            return response
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    def delete(self, ids: List[str], namespace: str = None) -> Dict[str, Any]:
        """Delete vectors by IDs."""
        index = self.get_index()
        namespace = namespace or self.namespace
        try:
            response = index.delete(ids=ids, namespace=namespace)
            return response
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise

    def create_namespace(self, namespace: str):
        """Create a new namespace (Pinecone handles this automatically on upsert)."""
        # Pinecone creates namespaces on first upsert, so this is a no-op
        logger.info(f"Namespace '{namespace}' will be created on first upsert.")

    def describe_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        index = self.get_index()
        try:
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Describe index stats failed: {e}")
            raise


# Convenience function to get a VectorStore instance
def get_vector_store(pinecone_client: Optional[Pinecone] = None) -> VectorStore:
    """Factory function to create VectorStore instance."""
    return VectorStore(pinecone_client)
