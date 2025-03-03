import numpy as np
from smolagents import tool
from config.model import HUGGINGFACE_MODEL_MAPPING, EmbeddingModel
from config.config import settings
from huggingface_hub import AsyncInferenceClient
import asyncio


# TODO: Switch out HuggingFace Inference API due to rate limits
@tool
async def calculate_text_similarity(text_1: str, text_2: str) -> float:
    """
    Computes the cosine similarity between two texts using an async Hugging Face embedding model.
    Remember to await this coroutine to get the values.

    Args:
        text_1: The first text input.
        text_2: The second text input.

    Returns:
        float: A similarity score between -1 and 1.
            - 1 means identical,
            - 0 means no similarity,
            - -1 means completely opposite.
    """
    client = AsyncInferenceClient(
        provider="hf-inference", api_key=settings.HUGGINGFACE_API_KEY
    )
    embedding_model = EmbeddingModel.BGE_M3

    async def embed_single_text(text: str):
        try:
            return await client.feature_extraction(
                model=HUGGINGFACE_MODEL_MAPPING[embedding_model], text=text
            )
        except Exception as e:
            print(f"Failed to generate embedding for text: {text} | Error: {e}")
            return None

    embeddings = await asyncio.gather(
        embed_single_text(text_1), embed_single_text(text_2)
    )

    if None in embeddings:
        return 0.0

    embedding_1, embedding_2 = embeddings

    if len(embedding_1) != len(embedding_2):
        raise ValueError("Embeddings must be of the same length.")

    # Cosine similarity
    numerator = np.dot(embedding_1, embedding_2)
    magnitude_a = np.linalg.norm(embedding_1)
    magnitude_b = np.linalg.norm(embedding_2)

    return (
        numerator / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0
    )
