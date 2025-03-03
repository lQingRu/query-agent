from huggingface_hub import InferenceClient
import numpy as np
from config.model import HUGGINGFACE_MODEL_MAPPING, EmbeddingModel
from config.config import settings
from smolagents import tool


# TODO: Switch out HuggingFace Inference API due to rate limits
# NOTE: Await not supported in smolagents code execution
@tool
def calculate_text_similarity(text_1: str, text_2: str) -> float:
    """
    Embed two texts and computes the cosine similarity between them.

    Args:
        text_1: The first text input.
        text_2: The second text input.

    Returns:
        float: A similarity score between -1 and 1.
    """
    client = InferenceClient(
        provider="hf-inference", api_key=settings.HUGGINGFACE_API_KEY
    )
    embedding_model = HUGGINGFACE_MODEL_MAPPING.get(EmbeddingModel.BGE_M3)

    try:
        embedding_1 = client.feature_extraction(model=embedding_model, text=text_1)
        embedding_2 = client.feature_extraction(model=embedding_model, text=text_2)
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return 0.0

    if len(embedding_1) != len(embedding_2):
        raise ValueError("Embeddings must be of the same length.")

    # Compute cosine similarity
    numerator = np.dot(embedding_1, embedding_2)
    denominator = np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)

    return numerator / denominator if denominator else 0.0
