import re
import unicodedata


def clean_markdown_page(text: str) -> str:
    """
    Limpa artefatos de conversão de PDF para Markdown preservando tabelas.
    """
    if not text:
        return ""

    # 1. Normalização Unicode (converte \xa0 em espaço comum, padroniza acentos)
    text = unicodedata.normalize("NFKC", text)

    # 2. Desfaz hifenização de final de linha (ex: ope-\nracional -> operacional)
    text = re.sub(r"([a-zA-ZÀ-ÿ]+)-\n\s*([a-zA-ZÀ-ÿ]+)", r"\1\2", text)

    # 3. Limpa tags HTML residuais das tabelas (<br>, <sup>, <sub>, etc.)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:sup|sub|b|i|strong|em|span|small)>", "", text, flags=re.IGNORECASE)

    # 4. Remove caracteres Unicode de área privada (ícones/glifos do PDF)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # 5. Normaliza espaços horizontais repetidos (mantendo quebras de linha)
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Limita quebras de linha seguidas a no máximo duas (\n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

