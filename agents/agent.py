from config.model import (
    HUGGINGFACE_MODEL_MAPPING,
    OLLAMA_MODEL_MAPPING,
    LargeLanguageModel,
)
from config.config import settings
from smolagents import HfApiModel, LiteLLMModel


def get_smolagents_agent(model: LargeLanguageModel):
    model_id = OLLAMA_MODEL_MAPPING.get(model)
    if model_id:
        return LiteLLMModel(model_id=model_id, api_base=settings.OLLAMA_URL)
    else:
        model_desc = HUGGINGFACE_MODEL_MAPPING.get(model)
        if not model_desc:
            raise ValueError(f"Invalid model specified: {model}")
        return HfApiModel(model_id=model_desc[0], provider=model_desc[1])
