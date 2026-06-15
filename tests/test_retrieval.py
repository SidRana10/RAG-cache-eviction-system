"""Retrieval logic tests"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from core.retrieval.retriever import retrieve_context


class TestRetrieveContext(unittest.TestCase):
    """Test suite for retrieve_context function."""

    @patch('core.retrieval.retriever.vector_db')
    @patch('core.retrieval.retriever._embedding_model')
    def test_retrieve_context_success(self, mock_embedding, mock_vector_db):
        """Test successful retrieval with query embedding."""
        # Setup
        query = "What is machine learning?"
        namespace = "docs"
        top_k = 5

        # Mock embedding
        mock_query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.embed_query.return_value = mock_query_embedding

        # Mock vector DB results
        mock_results = [
            {
                'id': 'doc1',
                'score': 0.95,
                'metadata': {
                    'text': 'ML is a subset of AI',
                    'source': 'wiki.pdf'
                }
            },
            {
                'id': 'doc2',
                'score': 0.87,
                'metadata': {
                    'text': 'Machine learning uses algorithms',
                    'source': 'textbook.pdf'
                }
            }
        ]
        mock_vector_db.query.return_value = mock_results
        
        # Execute
        result = retrieve_context(query, namespace, top_k)
        
        # Assert
        self.assertEqual(result, mock_results)
        mock_embedding.embed_query.assert_called_once_with(query)
        mock_vector_db.query.assert_called_once_with(
            vector=mock_query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

    @patch('core.retrieval.retriever.vector_db')
    def test_retrieve_context_invalid_namespace(self, mock_vector_db):
        """Test that invalid namespace raises error."""
        query = "test query"
        namespace = 'None'  # Invalid default
        
        with self.assertRaises(ValueError):
            retrieve_context(query, namespace, top_k=5)

    @patch('core.retrieval.retriever.vector_db')
    @patch('core.retrieval.retriever._embedding_model')
    def test_retrieve_context_embedding_failure(self, mock_embedding, mock_vector_db):
        """Test handling of embedding failure."""
        query = "test query"
        namespace = "valid_ns"

        # Mock embedding failure
        mock_embedding.embed_query.side_effect = RuntimeError("Embedding failed")

        with self.assertRaises(RuntimeError):
            retrieve_context(query, namespace)

    @patch('core.retrieval.retriever.vector_db')
    @patch('core.retrieval.retriever._embedding_model')
    def test_retrieve_context_vector_db_failure(self, mock_embedding, mock_vector_db):
        """Test handling of vector DB query failure."""
        query = "test query"
        namespace = "valid_ns"

        # Mock embedding success but DB failure
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vector_db.query.side_effect = Exception("DB connection failed")

        with self.assertRaises(Exception):
            retrieve_context(query, namespace)

    @patch('core.retrieval.retriever.vector_db')
    @patch('core.retrieval.retriever._embedding_model')
    def test_retrieve_context_custom_top_k(self, mock_embedding, mock_vector_db):
        """Test that custom top_k is passed correctly."""
        query = "test query"
        namespace = "valid_ns"
        top_k = 10

        mock_embedding.embed_query.return_value = [0.1, 0.2]
        mock_vector_db.query.return_value = []

        retrieve_context(query, namespace, top_k=top_k)

        # Verify top_k was passed correctly
        call_args = mock_vector_db.query.call_args
        self.assertEqual(call_args.kwargs['top_k'], top_k)


if __name__ == '__main__':
    unittest.main()