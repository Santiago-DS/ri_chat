from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
import typer

from ri_chat.engine.utils.chroma_collection import get_chroma_collection
from ri_chat.engine.utils.embeddings import load_embedding_model
from ri_chat.engine.utils.llm import (
    DEFAULT_LLM_MODEL,
    build_period_chain,
    build_rag_chain,
    extract_period,
    load_chat_model,
)
from ri_chat.engine.utils.retriever import retrieve_pages

app = typer.Typer()


def build_rag_components(
    model_name: str = DEFAULT_LLM_MODEL,
    max_new_tokens: int = 512,
) -> dict:
    """Constroi e retorna todos os componentes necessários para o pipeline RAG."""

    logger.info("Connecting to ChromaDB collection...")
    collection = get_chroma_collection()

    embedding_model = load_embedding_model()
    chat_model = load_chat_model(model_name=model_name, max_new_tokens=max_new_tokens)

    period_chain = build_period_chain(chat_model)
    rag_chain = build_rag_chain(chat_model)

    logger.success("RAG components successfully initialized!")

    return {
        "collection": collection,
        "embedding_model": embedding_model,
        "period_chain": period_chain,
        "rag_chain": rag_chain,
    }


def run_rag_pipeline(
    question: str,
    rag_components: dict,
    chat_history: list,
    n_chunks: int = 3,
) -> str:
    """Executa um único turno do pipeline RAG (roteamento temporal, busca e geração)."""

    question = question.encode("utf-8", errors="ignore").decode("utf-8").strip()

    # 1. Detecta o período automaticamente com base na pergunta + histórico
    period = extract_period(
        question=question,
        chat_history=chat_history,
        period_chain=rag_components["period_chain"],
    )
    logger.info(f"Detected period filter: {period}")

    # 2. Recupera as páginas completas no ChromaDB
    pages = retrieve_pages(
        query=question,
        collection=rag_components["collection"],
        embedding_model=rag_components["embedding_model"],
        n_chunks=n_chunks,
        period=period,
    )
    context_text = "\n\n---\n\n".join(page["content"] for page in pages)

    # 3. Gera a resposta com o LLM
    answer = rag_components["rag_chain"].invoke(
        {
            "chat_history": chat_history,
            "context": context_text,
            "question": question,
        }
    )

    # 4. Atualiza a memória da sessão
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer


@app.command()
def main(
    question: str = "Qual foi o resultado recorrente gerencial e o ROE no 4T2024?",
    model_name: str = DEFAULT_LLM_MODEL,
    n_chunks: int = 3,
    max_new_tokens: int = 512,
) -> None:
    """Executa uma única consulta de teste no pipeline RAG."""

    rag_components = build_rag_components(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    chat_history: list = []

    answer = run_rag_pipeline(
        question=question,
        rag_components=rag_components,
        chat_history=chat_history,
        n_chunks=n_chunks,
    )
    print(f"\n[Pergunta]: {question}")
    print(f"\n[RI Chat]:\n{answer}")


if __name__ == "__main__":
    app()
