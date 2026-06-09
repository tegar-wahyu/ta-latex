"""LlamaIndex + litellm wiring: LLM and embedding model factories."""

from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_llm(model: str) -> LiteLLM:
    """Return a LlamaIndex LLM backed by litellm."""
    return LiteLLM(model=model, temperature=0.1)


def get_embed_model(model: str):
    """Return embedding model - uses HuggingFace locally for huggingface/* models."""
    if model.startswith("huggingface/"):
        model_name = model.replace("huggingface/", "")
        return HuggingFaceEmbedding(model_name=model_name)
    return LiteLLMEmbedding(model_name=model)
