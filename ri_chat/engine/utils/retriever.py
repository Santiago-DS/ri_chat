import chromadb
from sentence_transformers import SentenceTransformer

from ri_chat.engine.utils.embeddings import generate_embeddings


def retrieve_pages(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    n_chunks: int = 3,
    period: str | None = None,
) -> list[dict]:
    """Busca os chunks mais relevantes e reconstrói as páginas completas (Parent-Page Retrieval)."""

    # gera embedding da pergunta do usuario
    query_emb = generate_embeddings(query, embedding_model, show_progress=False)

    # monta filtro opcional pelo metadado 'periodo' salvo no ChromaDB
    where_filter = {"periodo": period} if period else None

    # busca vetorial pelos chunks mais relevantes
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=n_chunks,
        where=where_filter,
    )

    # descobre as paginas unicas encontradas mantendo a ordem de relevancia
    seen_pages = []
    for meta in results["metadatas"][0]:
        page_key = (meta["periodo"], meta["tipo_doc"], meta["page_number"])
        if page_key not in seen_pages:
            seen_pages.append(page_key)

    # busca todos os chunks de cada pagina unica para reconstrui-la por completo
    pages = []
    for p_period, p_doc_type, p_page_num in seen_pages:
        page_data = collection.get(
            where={
                "$and": [
                    {"periodo": p_period},
                    {"tipo_doc": p_doc_type},
                    {"page_number": p_page_num},
                ]
            }
        )

        # ordena os chunks pelo chunk_index para reconstruir a leitura na ordem correta
        chunks_sorted = sorted(
            zip(page_data["metadatas"], page_data["documents"]),
            key=lambda item: item[0]["chunk_index"],
        )

        # remove os cabecalhos repetidos a partir do segundo chunk
        header = (
            f"[Relatório Itaú Unibanco | Período: {p_period} | "
            f"Demonstrações: {p_doc_type} | Página: {p_page_num}]"
        )
        clean_parts = [doc.replace(f"{header}\n", "") for _, doc in chunks_sorted]

        pages.append(
            {
                "periodo": p_period,
                "tipo_doc": p_doc_type,
                "page_number": p_page_num,
                "content": f"{header}\n" + "\n".join(clean_parts),
            }
        )

    return pages
