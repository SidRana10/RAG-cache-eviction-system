"""Module for Centralized configuration"""

import os
from dotenv import load_dotenv

# creating access to env variables
load_dotenv()


# Hugging face config
def huggingface_config():
    """serves configs for huggingface"""
    api_key = os.environ["HUGGINGFACE_API_KEY"]
    model_name = os.environ['HUGGINGFACE_MODEL_NAME']
    task = os.environ['HUGGINGFACE_TASK']
    embedding_model = os.environ['HUGGINGFACE_EMBEDDING_MODEL']
    eval_model = os.environ['HUGGINGFACE_EVAL_MODEL']
    return {
        'api key': api_key,
        'model': model_name,
        'task': task,
        'embedding model': embedding_model,
        'evaluation model': eval_model}


# pinecone config
def pinecone_config():
    """serves configs for pinecone vector database"""
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]
    namespace = os.environ["PINECONE_NAMESPACE"]
    index = os.environ['PINECONE_INDEX']
    return {
        'api key': api_key,
        'environment': environment,
        'namespace': namespace,
        'index': index}


# google config
def google_genai_config():
    """serves configs for google genai"""
    api_key = os.environ['GOOGLE_GENAI_API_KEY']

    return {
        'api key': api_key
    }


# benchmark constants
def benchmark_const():
    """serves values needed for benchmark module"""
    path = os.environ['PATH']
    filename = os.environ["FILENAME"]

    return {
        'path': path,
        'filename': filename
    }
