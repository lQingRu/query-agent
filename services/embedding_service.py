from huggingface_hub import InferenceClient
import ollama
from config.model import HUGGINGFACE_MODEL_MAPPING, OLLAMA_MODEL_MAPPING, EmbeddingModel
from config.config import settings
import numpy as np


def embed_query(text: str, embedding_model: EmbeddingModel):

    model_id = OLLAMA_MODEL_MAPPING.get(embedding_model)
    if model_id:
        return ollama.embed(model=model_id, input=text)
    else:
        model_desc = HUGGINGFACE_MODEL_MAPPING.get(embedding_model)
        if not model_desc:
            raise ValueError(f"Invalid model specified: {embedding_model}")
        client = InferenceClient(
            provider="hf-inference", api_key=settings.HUGGINGFACE_API_KEY
        )
        result = client.feature_extraction(
            model=HUGGINGFACE_MODEL_MAPPING[embedding_model],
            text=text,
        )
        return result


def cosine_similarity(embedding_1: str, embedding_2: str):
    numerator = np.dot(embedding_1, embedding_2)

    magnitude_a = np.linalg.norm(embedding_1)
    magnitude_b = np.linalg.norm(embedding_2)

    return numerator / (magnitude_a * magnitude_b)
