from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ri_chat.config import PROCESSED_DATA_DIR
from ri_chat.modeling.utils.embeddings import generate_embeddings, load_embedding_model
from ri_chat.modeling.utils.chroma_collection import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    create_chroma_collection,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "chunks.parquet",
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 500,
) -> None:
    """Gera os embeddings de todos os chunks e os indexa no ChromaDB."""

    # carregando chunks
    logger.info(f"Reading chunks from: {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Total chunks loaded: {len(df)}")

    # carregando o modelo e gerando as embeddings
    model = load_embedding_model()
    embeddings = generate_embeddings(
        texts=df["text_with_context"].tolist(),
        model=model,
    )

    # criando colecao
    logger.info(f"Creating collection '{collection_name}' at: {chroma_dir}")
    collection = create_chroma_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
    )

    # preparando insercao de metadados
    chunk_ids = df["chunk_id"].tolist()
    documents = df["text_with_context"].tolist()
    metadatas = df[
        ["periodo", "ano", "trimestre", "tipo_doc", "page_number", "chunk_index"]
    ].to_dict(orient="records")

    # 5. Inserindo em lotes (batches) para respeitar o limite do SQLite/ChromaDB
    total_chunks = len(df)
    for start_idx in tqdm(
        range(0, total_chunks, batch_size),
        desc="Indexing in ChromaDB",
    ):
        end_idx = min(start_idx + batch_size, total_chunks)

        collection.add(
            ids=chunk_ids[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx].tolist(),
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )

    logger.success(
        f"Indexing completed! Total documents in collection '{collection_name}': {collection.count()}"
    )


if __name__ == "__main__":
    app()
