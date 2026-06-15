"""Script to preprocess & index documents"""
import itertools
import uuid
from typing import Iterable


# custom modules
from ..core.utils.config import pinecone_config
from ..core.utils.startup import get_resources
from ..core.retrieval.embeddings import (
    load_and_split_document, create_embeddings)
from ..core.utils.logger import logger
from ..core.utils.helpers import timer, timeout
from ..core.utils.exceptions import BatchingError

# load resources needed for retrieval
resources = get_resources()
pinecone_configuration = pinecone_config()

# vector_db is needed for injesting,
# so we make it available at the module level
vector_db_client = resources["vector_db"]
namespace = pinecone_configuration["namespace"]
index_name = pinecone_configuration['index']


# turning vectors to iterables to reduce memory cost
@timer
def chunks(file, batch_size: int = 100) -> Iterable[tuple]:
    """serves data on demand streams to conserve memory usage"""
    # making the doc an iterable
    doc = load_and_split_document(file)
    iterable = iter(doc)

    # we are slicing the document into batches
    # and yielding its content one at a time to reduce memory usage
    chunk = tuple(itertools.islice(iterable, batch_size))
    counter = 1
    while chunk:
        yield chunk, counter
        counter += 1
        chunk = tuple(itertools.islice(iterable, batch_size))


# function to coalate records for upload
@timer
def _prepare_batch_records(batch, file_name):
    """prepares records for upload to vector database"""
    # generate embeddings to be used in similarity search
    embeddings = create_embeddings(batch)

    # checking that embedding function works properly
    # and does not return empty or wrong length results
    if len(embeddings) != len(batch):
        raise ValueError("Embeddings length does not match batch length")
    # generate ids for eacy referencing
    batch_id = str(uuid.uuid4())
    ids = [f"{batch_id}-{i}" for i in range(len(embeddings))]

    # metadata to used used to sort information
    metadatas = [{'source': file_name, 'text': text} for text in batch]

    # data to be stored on the vector database
    vectors = [
        (ids[i], embeddings[i], metadatas[i]) for i in range(len(embeddings))
          ]
    return vectors


# data injestion
# batch injestion
@timeout(180)
def ingest_in_batches(
        file=None, file_name=None, batch_size: int = 100):
    """injests vectors in batches to reduce memory usage"""
    # checks
    if file is None or file_name is None:
        logger.error("No file or file name provided for ingestion")
        raise ValueError(
            "check input: Needs both file and file name to continue"
        )
    logger.info("data ingestion by sequential batching initiated")
    try:
        # we process the data in chunks
        for batch, counter in chunks(file, batch_size):
            vectors = _prepare_batch_records(batch, file_name)
            vector_db_client.upsert(
                vectors=vectors,
                namespace=namespace)
            logger.info("batch %s complete", counter)
        logger.info("data ingestion by sequential batching complete")
        return 'Sequential batching complete'
    except ValueError as err:
        logger.error("ValueError occured, stopping the ingesti")
        raise ValueError(
            f"{err}: wrong input caused the failed ingestion") from err


# parallel ingesting
@timeout(120)
def ingest_in_parallel(
        file, file_name, thread_value: int = 30, batch_size: int = 100):
    """Submits upsert requests asynchronously for each chunk"""
    # checks
    if file is None or file_name is None:
        logger.error("No file or file name provided for ingestion")
        raise ValueError(
            "check input: Needs both file and file name to continue"
        )
    logger.info("data ingestion by parallel batching initiated")
    with vector_db_client.Index(
            index_name, pool_threads=thread_value
            ) as index:
        try:
            # we store the results of the asynchronous requests
            async_results = []
            # we process the data in chunks
            for batch, counter in chunks(file, batch_size):
                # we generate the embeddings and neccesary metadata for the batch
                # before submitting the request to the vector database
                # this makes it easy to access the data later
                vectors = _prepare_batch_records(batch, file_name)
                # we submit the request to the vector database asynchronously
                # this allows us to process the next batch while waiting for the current batch to be processed
                # this is more efficient than submitting the requests sequentially
                async_results.append(
                    index.upsert(
                        vectors=vectors,
                        namespace=namespace,
                        index_name=index_name,
                        async_req=True,
                    )
                )
                logger.info("batch %s complete", counter)

            # Collect results and check for exceptions
            for result in async_results:
                try:
                    result.get()
                except Exception as err:
                    logger.exception("Async upsert failed for one batch")
                    raise BatchingError(
                        "Failed to upsert batch asynchronously") from err
            logger.info("data ingestion by parallel batching successful")
            return "parallel batching complete"

        except ValueError as err:
            logger.error("ValueError occured, stopping the ingestion")
            raise ValueError(
                f"{err}: wrong input caused the failed ingestion") from err
