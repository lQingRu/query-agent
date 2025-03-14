import numpy as np
from config.model import EmbeddingModel, embedding_model


def calculate_text_similarity(text_1: str, text_2: str) -> float:
    """
    Embed two texts and computes the cosine similarity between them.

    Args:
        text_1: The first text input.
        text_2: The second text input.

    Returns:
        float: A similarity score between -1 and 1.
    """
    model = embedding_model(EmbeddingModel.BGE_M3)

    try:
        [text_1_embeddings, text_2_embeddings] = model.embed_documents(
            texts=[text_1, text_2]
        )
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return 0.0

    if len(text_1_embeddings) != len(text_2_embeddings):
        raise ValueError("Embeddings must be of the same length.")

    # Compute cosine similarity
    numerator = np.dot(text_1_embeddings, text_2_embeddings)
    denominator = np.linalg.norm(text_1_embeddings) * np.linalg.norm(text_2_embeddings)

    return numerator / denominator if denominator else 0.0
