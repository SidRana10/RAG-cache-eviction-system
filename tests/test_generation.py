"""LLM pipeline tests"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from core.generation.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test suite for RAGPipeline class."""

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    @patch('core.generation.rag_pipeline.retrieve_context')
    def test_generate_answer_success(self, mock_retrieve, mock_hf_class):
        """Test successful answer generation."""
        # Setup mocks
        mock_hf_instance = MagicMock()
        mock_pipeline = MagicMock()
        mock_hf_instance.pipeline = mock_pipeline
        mock_hf_class.return_value = mock_hf_instance
        
        # Mock retrieval results
        mock_docs = [
            {
                'id': 'doc1',
                'score': 0.95,
                'metadata': {
                    'text': 'Machine learning is powerful',
                    'source': 'doc1.pdf'
                }
            }
        ]
        mock_retrieve.return_value = mock_docs
        
        # Mock pipeline output
        mock_pipeline.return_value = [
            {'generated_text': 'Based on the context, ML is very useful'}
        ]
        
        # Execute
        pipeline = RAGPipeline()
        result = pipeline.generate_answer(
            query="What is machine learning?",
            namespace="docs",
            top_k=5
        )
        
        # Assert
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertEqual(len(result['sources']), 1)
        mock_retrieve.assert_called_once()

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    @patch('core.generation.rag_pipeline.retrieve_context')
    def test_generate_answer_empty_retrieval(self, mock_retrieve, mock_hf_class):
        """Test handling of empty retrieval results."""
        mock_hf_instance = MagicMock()
        mock_pipeline = MagicMock()
        mock_hf_instance.pipeline = mock_pipeline
        mock_hf_class.return_value = mock_hf_instance
        
        # No documents retrieved
        mock_retrieve.return_value = []
        mock_pipeline.return_value = [
            {'generated_text': 'No relevant context found.'}
        ]
        
        pipeline = RAGPipeline()
        result = pipeline.generate_answer(
            query="test",
            namespace="docs"
        )
        
        self.assertEqual(len(result['sources']), 0)

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    @patch('core.generation.rag_pipeline.retrieve_context')
    def test_generate_answer_pipeline_failure(self, mock_retrieve, mock_hf_class):
        """Test handling of pipeline generation failure."""
        mock_hf_instance = MagicMock()
        mock_pipeline = MagicMock()
        mock_hf_instance.pipeline = mock_pipeline
        mock_hf_class.return_value = mock_hf_instance
        
        mock_retrieve.return_value = [
            {'metadata': {'text': 'test', 'source': 'test.pdf'}}
        ]
        
        # Pipeline raises exception
        mock_pipeline.side_effect = RuntimeError("GPU memory error")
        
        pipeline = RAGPipeline()
        result = pipeline.generate_answer(
            query="test",
            namespace="docs"
        )
        
        # Should return error message gracefully
        self.assertIn("couldn't generate", result['answer'].lower())

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    @patch('core.generation.rag_pipeline.retrieve_context')
    def test_generate_answer_retrieval_failure(self, mock_retrieve, mock_hf_class):
        """Test handling of retrieval failure."""
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        # Retrieval fails
        mock_retrieve.side_effect = Exception("Connection error")
        
        pipeline = RAGPipeline()
        
        with self.assertRaises(Exception):
            pipeline.generate_answer(
                query="test",
                namespace="docs"
            )

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    def test_build_context_with_multiple_docs(self, mock_hf_class):
        """Test context building from multiple documents."""
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        docs = [
            {
                'metadata': {'text': 'AI is intelligent', 'source': 'doc1.pdf'}
            },
            {
                'metadata': {'text': 'ML is learning', 'source': 'doc2.pdf'}
            }
        ]
        
        pipeline = RAGPipeline()
        context = pipeline._build_context(docs)
        
        self.assertIn('AI is intelligent', context)
        self.assertIn('ML is learning', context)
        self.assertIn('doc1.pdf', context)
        self.assertIn('doc2.pdf', context)

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    def test_extract_sources(self, mock_hf_class):
        """Test source extraction from documents."""
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        docs = [
            {
                'score': 0.95,
                'metadata': {'text': 'Test text', 'source': 'source1.pdf'}
            },
            {
                'score': 0.87,
                'metadata': {'text': 'Another text', 'source': 'source2.pdf'}
            }
        ]
        
        pipeline = RAGPipeline()
        sources = pipeline._extract_sources(docs)
        
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]['score'], 0.95)
        self.assertEqual(sources[0]['source'], 'source1.pdf')
        self.assertEqual(sources[1]['source'], 'source2.pdf')

    @patch('core.generation.rag_pipeline.HUGGINGFACE')
    def test_build_prompt_format(self, mock_hf_class):
        """Test prompt building format."""
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        pipeline = RAGPipeline()
        prompt = pipeline._build_prompt(
            query="What is AI?",
            context="AI is artificial intelligence"
        )
        
        self.assertIn("What is AI?", prompt)
        self.assertIn("AI is artificial intelligence", prompt)
        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)


if __name__ == '__main__':
    unittest.main()