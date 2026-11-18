from pathlib import Path

import chromadb
from loguru import logger

from ri_chat.config import MODELS_DIR

DEFAULT_CHROMA_DIR = MODELS_DIR / "vector_store" / "chromadb"
DEFAULT_COLLECTION_NAME = "itau_ri_chunks"


def create_chroma_collection(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    """Cria (ou recria do zero) uma coleção no ChromaDB persistente."""

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # caso a colecao ja exista, remove para criar do zero
    try:
        client.delete_collection(collection_name)
        logger.info(f"Previous collection '{collection_name}' removed.")
    except Exception:  # noqa: BLE001, S110
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def get_chroma_collection(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    """Carrega uma coleção já existente do ChromaDB para consultas."""

    client = chromadb.PersistentClient(path=str(chroma_dir))

    return client.get_collection(name=collection_name)

