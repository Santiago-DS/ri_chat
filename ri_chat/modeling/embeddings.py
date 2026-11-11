from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cuda",
) -> SentenceTransformer:
    """Carrega o modelo SentenceTransformer na GPU."""

    if not torch.cuda.is_available():
        logger.error("Sem GPU disponível!")

    logger.info(f"Loading embedding model '{model_name}' on '{device}'...")

    return SentenceTransformer(model_name, device=device)


def generate_embeddings(
    texts: list[str] | str,
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Gera vetores normalizados (L2) para uma string ou lista de textos."""

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )

    return embeddings
