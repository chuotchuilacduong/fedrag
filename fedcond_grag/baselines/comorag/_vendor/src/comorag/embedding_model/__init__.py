
from .base import EmbeddingConfig, BaseEmbeddingModel
from .BGEEmbedding import BGEEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "None"):
    if "bge-" in embedding_model_name.lower():
        return BGEEmbeddingModel
    elif "text-embedding-3-small" in embedding_model_name:
        return OpenAIEmbeddingModel
    else:
        # NOTE (fedrag fork): upstream's default branch was `return` with no
        # value -- the log message says "using BGEEmbeddingModel as
        # default" but it actually returned None, which would crash the
        # first time this factory is called with any name that isn't
        # "bge-*" or "text-embedding-3-small" (e.g. this project's usual
        # all-MiniLM-L6-v2). BGEEmbeddingModel itself is a generic
        # AutoModel/AutoTokenizer wrapper (see BGEEmbedding.py), not
        # actually BGE-specific, so returning it here matches upstream's
        # stated intent.
        logger.info(f"Unknown embedding model name: {embedding_model_name}, using BGEEmbeddingModel as default")
        return BGEEmbeddingModel 