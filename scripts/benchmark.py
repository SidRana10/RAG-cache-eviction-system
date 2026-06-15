"""Handles Performance evaluation"""
import time
from pathlib import Path
import pandas as pd
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from ..core.utils.config import benchmark_const, google_genai_config
from ..core.utils.helpers import timeout, export_to_csv
from ..core.utils.logger import logger

# load required variables
CONSTANTS = benchmark_const()
GOOGLE_VARIABLES = google_genai_config()

# benchmark constants
FILENAME = CONSTANTS['filename'] + '.csv'
PATH = Path(CONSTANTS['path'])
FILE = PATH / FILENAME

# data for export
SAVE_PATH = Path.home() / "data/processed"
SAVE_NAME = "evaluation_result"

# google variables
google_api_key = GOOGLE_VARIABLES['api key']


# load the data in chunks
def load_file(batch_size: int = 2):
    """Serves data in batches to conserve memory usage"""
    data_iter = pd.read_csv(FILE, chunksize=batch_size)
    for chunk in data_iter:
        yield chunk
        # rate limiting: only invoke twice per minute
        time.sleep(30)


# model to be invoked for evaluation
@timeout(30)
@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(5))
def load_judge(model: str = "gemini-2.5-pro"):
    """Loads model from Google for evaluation"""
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=google_api_key
    )


# eval function to carry out evaluation
def google_evaluator():
    """carries out evaluations"""
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=load_judge()
        )


def _safe_score(eval_result):
    """Safely normalize score into 1/0/None."""
    # score payload can come back in different formats
    score = eval_result.get("score")
    if isinstance(score, bool):
        return 1 if score else 0
    if isinstance(score, str):
        lowered = score.strip().lower()
        if lowered == "true":
            return 1
        if lowered == "false":
            return 0
    if score in (0, 1):
        return score
    return None


# evalutaion pipeline to be activated on push
def evaluate_correctness():
    """Evaluate correctness of responses using LLM as judge"""
    try:
        # make sure export folder exists before we start writing
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        # initialize once so we dont recreate judge for every row
        evaluator = google_evaluator()
    except Exception as exc:
        logger.exception("Failed to initialize evaluator: %s", exc)
        return

    try:
        # process in chunks to avoid loading full file into memory
        for batch in load_file():
            for _, row in batch.iterrows():
                try:
                    # evaluate each row’s response
                    query = row["query"]
                    response = row["answer"]
                    reference = row["reference"]

                    # run correctness eval
                    eval_result = evaluator(
                        inputs=query,
                        outputs=response,
                        reference_outputs=reference,
                    )
                    score = _safe_score(eval_result)
                    result = [query, response, reference, score]
                    export_to_csv(result, SAVE_PATH, SAVE_NAME, evaluation=True)
                except KeyError as exc:
                    # row shape is not what we expected
                    logger.error("Missing expected column in row: %s", exc)
                except Exception as exc:
                    # dont stop full run because one row failed
                    logger.exception("Row evaluation failed: %s", exc)
    except FileNotFoundError as exc:
        # input csv path is invalid or file is missing
        logger.exception("Benchmark input file not found: %s", exc)
    except pd.errors.EmptyDataError as exc:
        # file exists but has no parsable rows
        logger.exception("Benchmark input file is empty: %s", exc)
    except Exception as exc:
        # catch-all for anything unexpected in outer loop
        logger.exception("Benchmark evaluation failed: %s", exc)


if __name__ == "__main__":
    evaluate_correctness()