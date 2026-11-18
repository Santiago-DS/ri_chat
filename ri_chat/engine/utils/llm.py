import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from ri_chat.config import PROJ_ROOT

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPTS_DIR = PROJ_ROOT / "ri_chat" / "engine" / "prompts"
PERIOD_EXTRACTOR_PROMPT = (PROMPTS_DIR / "period_extractor_prompt.txt").read_text(encoding="utf-8")
SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def load_chat_model(
    model_name: str = DEFAULT_LLM_MODEL,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> ChatHuggingFace:
    """Carrega o LLM quantizado em 4-bit (NF4) e o empacota no ChatHuggingFace do LangChain."""

    if not torch.cuda.is_available():
        logger.error("No GPU available!")

    logger.info(f"Loading 4-bit quantized LLM '{model_name}' on '{device}'...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
    )

    txt_gen_pipe = pipeline(
        task="text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    hf_pipeline = HuggingFacePipeline(pipeline=txt_gen_pipe)
    return ChatHuggingFace(llm=hf_pipeline)


def build_period_chain(chat_model: ChatHuggingFace) -> Runnable:
    """Constrói a chain responsável por identificar o período (XT20YY) da pergunta."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PERIOD_EXTRACTOR_PROMPT),
            ("human", "HISTÓRICO DA CONVERSA:\n{history_text}\n\nPERGUNTA ATUAL:\n{question}"),
        ]
    )
    return prompt | chat_model | StrOutputParser()


def build_rag_chain(chat_model: ChatHuggingFace) -> Runnable:
    """Constrói a chain principal do RAG com memória conversacional."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "CONTEXTO:\n{context}\n\nPERGUNTA:\n{question}"),
        ]
    )
    return prompt | chat_model | StrOutputParser()


def extract_period(
    question: str,
    chat_history: list,
    period_chain: Runnable,
) -> str | None:
    """Extrai e valida o período fiscal ('XT20YY') a partir da pergunta e do histórico."""

    history_text = (
        "\n".join(f"{msg.type}: {msg.content}" for msg in chat_history) or "Sem histórico."
    )
    raw_period = period_chain.invoke({"question": question, "history_text": history_text}).strip()

    return raw_period if re.match(r"^[1-4]T202[4-6]$", raw_period) else None
