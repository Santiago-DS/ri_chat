from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ri_chat.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "pages_extracted.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "chunks.parquet",
    min_chars: int = 200,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    # carregando as paginas
    logger.info(f"Lendo páginas de: {input_path}")
    df = pd.read_parquet(input_path)

    # excluindo paginas de capa e transicao
    df_clean = df[(df["page_number"] > 1) & (df["char_count"] >= min_chars)].copy()
    logger.info(
        f"Páginas antes do filtro: {len(df)} | Após filtro: {len(df_clean)} | Páginas Removidas: {len(df) - len(df_clean)}"
    )

    # configurando o text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # fatiando cada pagina e injetando metadados no texto
    chunks_records = []

    for row in tqdm(df_clean.itertuples(index=False), total=len(df_clean), desc="Gerando chunks"):
        # criando os chunks
        page_chunks = splitter.split_text(row.text)

        # injetando os cabecalhos de metadados no texto
        periodo = f"{row.trimestre}T{row.ano}"
        header = (
            f"[Relatório Itaú Unibanco | Período: {periodo} | "
            f"Demonstrações: {row.tipo_doc} | Página: {row.page_number}]"
        )

        # agregando o resultado
        for idx, chunk_text in enumerate(page_chunks):
            chunks_records.append(
                {
                    "chunk_id": f"{periodo}_{row.tipo_doc}_p{row.page_number}_c{idx}",
                    "periodo": periodo,
                    "ano": row.ano,
                    "trimestre": row.trimestre,
                    "tipo_doc": row.tipo_doc,
                    "page_number": row.page_number,
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                    "text_with_context": f"{header}\n{chunk_text}",
                }
            )

    # salvando em parquet na camada processed
    df_chunks = pd.DataFrame(chunks_records)
    df_chunks.to_parquet(output_path, index=False)

    logger.success(
        f"Chunking concluído! {len(df_chunks)} chunks gerados e salvos em: {output_path}"
    )


if __name__ == "__main__":
    app()
