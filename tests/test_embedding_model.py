from config.model import EmbeddingModel
from services.embedding_service import cosine_similarity, embed_query

# NOTE: Temporarily used to generate different embeddings and differences between queries through tests

BROAD_SPECIFIC_1 = [
    "Any recent news on Ukraine and US?",
    "What military aid packages has the US provided to Ukraine in 2024 to support the war against Russia?",
]
DIFFERENT_QUESTIONS_1 = [
    "How have US sanctions on Russia affected Ukraine's economy in 2024?",
    "What is the US public’s opinion on continued military aid to Ukraine in 2024, according to major news sources?",
]


def test_e5_broad_specific():
    orig_embedding = embed_query(
        BROAD_SPECIFIC_1[0], EmbeddingModel.MULTILINGUAL_E5_LARGE
    )
    new_embedding = embed_query(
        BROAD_SPECIFIC_1[1], EmbeddingModel.MULTILINGUAL_E5_LARGE
    )
    print(cosine_similarity(orig_embedding, new_embedding))  # 0.8826904
    assert False


def test_e5_different_questions():
    orig_embedding = embed_query(
        DIFFERENT_QUESTIONS_1[0], EmbeddingModel.MULTILINGUAL_E5_LARGE
    )
    new_embedding = embed_query(
        DIFFERENT_QUESTIONS_1[1], EmbeddingModel.MULTILINGUAL_E5_LARGE
    )
    print(cosine_similarity(orig_embedding, new_embedding))  # 0.8881358
    assert False


def test_bge_m3_broad_specific():
    orig_embedding = embed_query(BROAD_SPECIFIC_1[0], EmbeddingModel.BGE_M3)
    new_embedding = embed_query(BROAD_SPECIFIC_1[1], EmbeddingModel.BGE_M3)
    print(cosine_similarity(orig_embedding, new_embedding))  # 0.63021034
    assert False


def test_bge_m3_different_questions():
    orig_embedding = embed_query(DIFFERENT_QUESTIONS_1[0], EmbeddingModel.BGE_M3)
    new_embedding = embed_query(DIFFERENT_QUESTIONS_1[1], EmbeddingModel.BGE_M3)
    print(cosine_similarity(orig_embedding, new_embedding))  # 0.70827377
    assert False
