from pathlib import Path
import re

import pymupdf

from ri_chat.utils.data_cleaning import clean_markdown_page


def extract_data_from_pdf(path_arq: Path) -> list[dict]:
    """Extrai todas as páginas de um PDF em formato Markdown estruturado."""
    with pymupdf.open(path_arq) as doc:
        pages_data = doc.to_markdown(
            page_chunks=True,
            header=False,
            footer=False,
            use_ocr=False,
        )
    return pages_data


def extract_metadata_from_text(text_md: str) -> dict:
    """
    Extrai trimestre, ano e tipo de demonstração a partir do Markdown da capa.
    """
    padrao_periodo = r"([1-4])\s*(?:º|°|<sup>o</sup>)?\s*trimestre\s+de\s+(20\d{2})"
    match = re.search(padrao_periodo, text_md, re.IGNORECASE)

    if match:
        trimestre = int(match.group(1))
        ano = int(match.group(2))
    else:
        trimestre, ano = None, None

    if re.search(r"condensad[ao]s", text_md, re.IGNORECASE):
        tipo_doc = "condensadas"
    elif re.search(r"complet[ao]s", text_md, re.IGNORECASE):
        tipo_doc = "completas"
    else:
        tipo_doc = "desconhecido"

    return {
        "trimestre": trimestre,
        "ano": ano,
        "tipo_doc": tipo_doc,
    }


def process_pdf_pages(pdf_path: Path, min_chars: int = 80) -> list[dict]:
    """
    Extrai, limpa e anexa os metadados da capa a cada página útil de um PDF.
    Retorna uma lista de dicionários (1 dicionário por página).
    """
    pages_data = extract_data_from_pdf(pdf_path)
    cover_meta = extract_metadata_from_text(pages_data[0]["text"])
    page_count = len(pages_data)

    records = []
    for page in pages_data:
        cleaned_text = clean_markdown_page(page["text"])

        # Ignora páginas vazias ou muito curtas
        if len(cleaned_text) < min_chars:
            continue

        records.append(
            {
                **cover_meta,
                "page_number": page["metadata"]["page_number"],
                "page_count": page_count,
                "char_count": len(cleaned_text),
                "text": cleaned_text,
            }
        )

    return records
