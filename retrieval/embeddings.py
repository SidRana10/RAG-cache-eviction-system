"""Handles Embedding generation"""
from typing import List, Document, tuple
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)

# custom modules
from ..utils.config import huggingface_config
from ..utils.logger import logger
from ..utils.helpers import timer, returns
from ..utils.exceptions import EmbeddingError

# conifigurations
embedding_model = huggingface_config()['embedding model']


# document spliter functions
@timer
def semantic_split(
    model: str = embedding_model,
    breakpoint_threshold_type: str = 'gradient',
    breakpoint_threshold_amount: float = 0.8
):
    """splits documents by semantic meaning"""
    # Initialize the SemanticChunker
    text_splitter = SemanticChunker(
        embeddings=model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    return text_splitter


@timer
def recursive_split(
        separators: List[str] = [],
        chunk_size: int = 1000,
        chunk_overlap: int = 200):

    """splits documents by characters recursively"""
    if separators == []:
        separators = ['\n\n', '\n', ' ', '']

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    return text_splitter


@timer
def load_and_split_document(
        file_path: str,
        split_type: str = 'semantic_split'):
    """Loads a PDF document, splits it into chunks, \
        and generates embeddings for each chunk."""
    # Validate file type
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid file type: {file_path}.\
                          Only PDF files are supported.")

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    try:
        if split_type == 'semantic_split':
            doc_splitter = semantic_split()
        elif split_type == 'recursive_split':
            doc_splitter = recursive_split()
        else:
            raise ValueError(
                f"Invalid split type: {split_type}.\
                    Must be 'semantic_split' or 'recursive_split'.")
        chunks = doc_splitter.split_documents(documents)
        return chunks
    except Exception as err:
        raise ValueError(
                f"Invalid split type: {split_type}. \
                    Must be 'semantic_split' or 'recursive_split'.") from err


# query splitter functions
@timer
def character_split(
        separator: str = ' ',
        chunk_size: int = 24, chunk_overlap: int = 3):
    """splits queries by characters"""
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter


@timer
def token_split(
        chunk_size: int = 10,
        chunk_overlap: int = 3,
        model_name: str = 'gpt-4o-mini'):
    """splits queries by tokens"""

    # ensuring token counts are accurate
    encoding = tiktoken.encoding_for_model(model_name)

    # creating splitter
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding.name
    )
    return text_splitter


@timer
def load_and_split_query(query: str, split_type: str = 'character_split'):
    """splits a query into chunks"""
    try:
        if split_type == 'character_split':
            text_splitter = character_split(separator=' ')
        elif split_type == 'token_split':
            text_splitter = token_split()
        else:
            raise ValueError(f"Unsupported split type: {split_type}")

        chunks = text_splitter.split_text(query)
        return chunks
    except Exception as err:
        raise ValueError(
            f"Error while splitting query with {split_type}: {err}"
        ) from err


@timer
@returns(list)
def create_embeddings(
        chunks: tuple[Document],
        model: str = embedding_model):
    """creates embeddings from either queries or documents"""
    try:
        embed_model = HuggingFaceEmbeddings(model_name=model)
        embeddings = embed_model.embed_documents(chunks)
        logger.info("embeddings successfully created")
        return embeddings
    except Exception as err:
        logger.info("embedding failed")
        raise EmbeddingError(f"{err}: could not embed successfully") from err
