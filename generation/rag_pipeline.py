"""Combines retriever and generator for RAG pipeline"""

from typing import List, Dict, Any
from .llm import HUGGINGFACE
from ..retrieval.retriever import retrieve_context
from ..utils.logger import logger
from ..utils.helpers import timer
from ..utils.config import pinecone_config


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline combining retrieval and generation."""
    VECTOR_DB_RESOURCES = pinecone_config()
    NAMESPACE = VECTOR_DB_RESOURCES['namespace']

    def __init__(self):
        """Initialize the RAG pipeline with model and retriever."""
        self.generator = HUGGINGFACE()
        self.generator.load_model()

    @timer
    def generate_answer(
            self, query: str, namespace: str = RAGPipeline().NAMESPACE,
            top_k: int = 5) -> Dict[str, Any]:
        """Generate an answer using retrieval-augmented generation.

        Args:
            query: The user's question
            namespace: The vector store namespace to search
            top_k: Number of top similar documents to retrieve

        Returns:
            Dict containing 'answer' and 'sources'
        """
        try:
            # Step 1: Retrieve relevant context
            retrieved_docs = retrieve_context(query, namespace, top_k)

            # Step 2: Build prompt with context
            context_text = self._build_context(retrieved_docs)
            prompt = self._build_prompt(query, context_text)

            # Step 3: Generate answer
            answer = self._generate_response(prompt)

            # Step 4: Prepare sources
            sources = self._extract_sources(retrieved_docs)

            return {
                'answer': answer,
                'sources': sources
            }

        except Exception as exec:
            logger.error("RAG pipeline failed: %s", exec)
            raise

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            text = metadata.get('text', doc.get('text', ''))
            source = metadata.get('source', f'Document {i}')
            context_parts.append(f"[Source {i} - {source}]: {text}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the generator."""
        prompt = f"""Based on the following context, answer the question accurately. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def _generate_response(self, prompt: str) -> str:
        """Generate response using the loaded model."""
        try:
            if self.generator.pipeline is None:
                logger.error("Pipeline not loaded")
                return "Sorry, the model failed to load."

            # Call pipeline with task-appropriate parameters
            # Most text-generation tasks support these common parameters
            outputs = self.generator.pipeline(
                prompt,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )

            # Handle different output formats from transformers
            if isinstance(outputs, list) and len(outputs) > 0:
                if isinstance(outputs[0], dict):
                    # Standard format: [{'generated_text': '...'}]
                    return outputs[0].get('generated_text', str(outputs)).strip()
                else:
                    # Fallback for other formats
                    return str(outputs[0]).strip()
            else:
                return str(outputs).strip() if outputs else ""
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return "Sorry, I couldn't generate a response at this time."

    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents."""
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            sources.append({
                'text': metadata.get('text', ''),
                'source': metadata.get('source', 'Unknown'),
                'score': doc.get('score', 0.0)
            })
        return sources
