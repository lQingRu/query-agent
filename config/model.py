from enum import Enum


class EmbeddingModel(Enum):
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"
    BGE_M3 = "bge-m3"


class LargeLanguageModel(Enum):
    # Reasoning models
    DEEPSEEK_R1_1_5b = "deepseek-r1:1.5b"
    PHI_4 = "phi-4"


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
    LargeLanguageModel.DEEPSEEK_R1_1_5b: "ollama/deepseek-r1:latest",
    LargeLanguageModel.PHI_4: "ollama/phi4:latest",
    EmbeddingModel.BGE_M3: "ollama/bge-m3:latest",
}
