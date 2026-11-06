from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ri_chat.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from ri_chat.utils import data_manipulation as dm

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "pages_extracted.parquet",
    min_chars: int = 80,
):
    pdf_files = sorted(input_dir.glob("*.pdf"))
    logger.info(f"Processando {len(pdf_files)} arquivos PDF...")

    all_pages = []
    for pdf_path in tqdm(pdf_files, desc="Extraindo PDFs"):
        pdf_records = dm.process_pdf_pages(pdf_path, min_chars=min_chars)
        all_pages.extend(pdf_records)

    df = pd.DataFrame(all_pages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.success(f"Concluído! {len(df)} páginas salvas em {output_path}")


if __name__ == "__main__":
    app()
