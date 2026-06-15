# Retrieval-Augmented Generation (RAG) Chat System

A FastAPI-based Retrieval-Augmented Generation chatbot that combines a Pinecone vector database with Hugging Face language models to answer questions grounded in your own documents. The system ingests PDF documents, embeds and indexes them, retrieves relevant context at query time, and generates accurate, source-cited answers.

## Features

- **FastAPI backend** with an async lifecycle that initializes external resources (vector DB) on startup
- **RAG pipeline** that retrieves relevant document chunks and feeds them to a Hugging Face generation model
- **Pinecone vector store** integration for similarity search, upserts, and deletes
- **Flexible document chunking**: semantic chunking (via `SemanticChunker`) or recursive character splitting
- **Query chunking**: character-based or token-based splitting using `tiktoken`
- **Batch data ingestion** with sequential and parallel (multi-threaded) upsert strategies
- **Automated LLM-as-judge benchmarking** using Google Gemini and the `openevals` correctness prompt
- **Resilience built in**: retries with exponential backoff (`tenacity`) on model loading, DB connections, and evaluations
- **Structured logging** with rotating file handlers and HTTP request timing middleware
- **Unit tests** for retrieval and generation logic using `unittest` and mocks
- **Dockerized deployment** with a multi-stage build for a lean runtime image

## Architecture

User Query

│

▼

FastAPI /v1/chat endpoint

│

▼

RAG Pipeline

├── Embed query (HuggingFace embeddings)

├── Retrieve top-k chunks (Pinecone vector store)

├── Build context + prompt

└── Generate answer (HuggingFace text-generation pipeline)

│

▼

Response (answer + sources)

│

▼

Background task → log interaction to CSV (for benchmarking)

Design diagrams are available in the `Design plans/` directory.

## Project Structure

├── api/

│   ├── server.py              # FastAPI app factory & entrypoint

│   ├── routes/

│   │   └── chatbot.py          # /chat and /services/health endpoints

│   └── middleware/

│       ├── schema.py           # Pydantic request/response models

│       └── auth.py             # Authentication (placeholder)

├── core/

│   ├── generation/

│   │   ├── llm.py               # Hugging Face model provider

│   │   └── rag_pipeline.py       # Combines retrieval + generation

│   ├── retrieval/

│   │   ├── embeddings.py         # Document/query chunking & embedding

│   │   ├── retriever.py          # Context retrieval logic

│   │   └── vectore_store.py      # Pinecone vector store wrapper

│   └── utils/

│       ├── config.py             # Centralized env-based configuration

│       ├── startup.py            # Resource initialization (Pinecone)

│       ├── helpers.py            # Decorators (timer, timeout, returns), CSV export

│       ├── logger.py             # Logging setup & HTTP middleware

│       └── exceptions.py         # Custom exceptions

├── scripts/

│   ├── ingest_data.py            # Batch document ingestion into Pinecone

│   └── benchmark.py              # LLM-as-judge correctness evaluation

├── tests/

│   ├── test_retrieval.py

│   ├── test_generation.py

│   └── test_api.py

├── Design plans/                  # Architecture diagrams

├── Dockerfile

├── requirements.txt

└── config.yml


## Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account, API key, and index
- A [Hugging Face](https://huggingface.co/) account/API key (for model and embedding access)
- A [Google AI](https://ai.google.dev/) API key (for the benchmark/evaluation judge model)
- Docker (optional, for containerized deployment)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Retrieval-Augmented-Generation-chat-system
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with the following variables:

```env
# Hugging Face
HUGGINGFACE_API_KEY=your_huggingface_api_key
HUGGINGFACE_MODEL_NAME=facebook/bart-large-cnn
HUGGINGFACE_TASK=summarization
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_EVAL_MODEL=google/flan-t5-small

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_NAMESPACE=chatbot
PINECONE_INDEX=your_index_name

# Google GenAI (used for benchmarking/evaluation)
GOOGLE_GENAI_API_KEY=your_google_genai_api_key

# Benchmarking
PATH=/path/to/save/interaction/logs
FILENAME=interactions
```

## Usage

### Run the API server

```bash
uvicorn api.server:server --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint              | Description                                  |
|--------|-----------------------|-----------------------------------------------|
| POST   | `/v1/chat`             | Submit a query and receive a generated answer with sources |
| GET    | `/v1/services/health`  | Check whether core services (vector DB) are initialized |

**Example request:**

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is retrieval-augmented generation?"}'
```

**Example response:**

```json
{
  "response": "Retrieval-augmented generation combines...",
  "sources": [
    {
      "text": "RAG is a technique that...",
      "source": "paper.pdf",
      "score": 0.91
    }
  ]
}
```

### Ingest documents

Use `scripts/ingest_data.py` to chunk, embed, and upsert PDF documents into your Pinecone index. The script supports both sequential and parallel batch ingestion to manage memory and throughput.

```python
from scripts.ingest_data import ingest_in_batches, ingest_in_parallel

# Sequential
ingest_in_batches(file="path/to/document.pdf", file_name="document.pdf")

# Parallel (multi-threaded upserts)
ingest_in_parallel(file="path/to/document.pdf", file_name="document.pdf")
```

### Run benchmarks

`scripts/benchmark.py` evaluates the correctness of generated responses against reference answers using a Gemini model as an LLM judge, reading from the interaction log produced by the API and writing scored results to `~/data/processed/evaluation_result.csv`.

```bash
python -m scripts.benchmark
```

## Testing

Run the test suite with:

```bash
pytest
```

Tests cover retrieval logic (`test_retrieval.py`) and the RAG generation pipeline (`test_generation.py`), using mocked vector store and model dependencies.

## Docker

Build and run the containerized service:

```bash
docker build -t rag-chat-system .
docker run -p 8501:8501 --env-file .env rag-chat-system
```

The Dockerfile uses a multi-stage build (builder + slim runtime) and runs as a non-root user. Default Hugging Face and Pinecone configuration values are baked in via `ENV` but can be overridden at runtime.

## Tech Stack

- **API**: FastAPI, Uvicorn
- **LLM & Embeddings**: Hugging Face Transformers, LangChain
- **Vector Database**: Pinecone
- **Evaluation**: Google Gemini, OpenEvals, RAGAS
- **Resilience**: Tenacity (retry with exponential backoff)
- **Data**: Pandas, DVC (for data versioning)






























