from enum import Enum
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from config.config import settings


class EmbeddingModel(Enum):
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"
    BGE_M3 = "bge-m3"


class LargeLanguageModel(Enum):
    # Reasoning models
    DEEPSEEK_R1_1_5b = "deepseek-r1:1.5b"
    PHI_4 = "phi-4"
    GEMMA_3_1b = "gemma3:1b"


HUGGINGFACE_MODEL_MAPPING = {
    EmbeddingModel.MULTILINGUAL_E5_LARGE: [
        "intfloat/multilingual-e5-large-instruct",
        "hf-inference",
    ],
    EmbeddingModel.BGE_M3: ["BAAI/bge-m3", "hf-inference"],
    LargeLanguageModel.DEEPSEEK_R1_1_5b: [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "hf-inference",
    ],
}

OLLAMA_MODEL_MAPPING = {
    LargeLanguageModel.DEEPSEEK_R1_1_5b: "deepseek-r1:latest",
    LargeLanguageModel.PHI_4: "phi4:latest",
    EmbeddingModel.BGE_M3: "bge-m3:latest",
    LargeLanguageModel.GEMMA_3_1b: "gemma3:1b",
}


def llm(model: LargeLanguageModel, temperature: float = 0.01) -> ChatOllama:
    model_id = OLLAMA_MODEL_MAPPING.get(model)
    if model_id:
        return ChatOllama(
            model=model_id,
            temperature=temperature,
            base_url=settings.OLLAMA_URL,
        )
    else:
        raise Exception(f"Model {model} not supported")


def embedding_model(model: EmbeddingModel) -> OllamaEmbeddings:
    model_id = OLLAMA_MODEL_MAPPING.get(model)
    if model_id:
        return OllamaEmbeddings(model=model_id, base_url=settings.OLLAMA_URL)
    else:
        raise Exception(f"Model {model} not supported")
