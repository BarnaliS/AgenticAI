from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI


QNA_SYSTEM_MESSAGE = """You are an AI assistant that answers questions based only on the provided context.

Rules:
- Use only the information from the retrieved context.
- If the answer is not in the context, say "I don't know".
- Do not make up information.
- Be concise and clear.
- If relevant, summarize the information.

Answer the user's question based on the context."""

QNA_USER_MESSAGE_TEMPLATE = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 5000
    top_p: float = 0.95
    frequency_penalty: float = 1.2
    stop_sequences: tuple[str, ...] = ("INST",)
    # Safety valve so we don't blow up the prompt budget.
    max_context_chars: int = 20_000


def get_chat_model(config: GenerationConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        stop_sequences=list(config.stop_sequences),
    )


def build_qa_prompt(question: str, context: str, config: GenerationConfig) -> str:
    if not context.strip():
        context = "No relevant context was retrieved."

    trimmed_context = context.strip()
    if config.max_context_chars and len(trimmed_context) > config.max_context_chars:
        trimmed_context = trimmed_context[: config.max_context_chars] + "..."

    user_message = QNA_USER_MESSAGE_TEMPLATE.format(context=trimmed_context, question=question)
    # Notebook uses a Llama-style [INST] wrapper.
    return f"[INST]{QNA_SYSTEM_MESSAGE}\n\nuser: {user_message}\n[/INST]"


def answer_question(
    question: str,
    context: str,
    config: GenerationConfig,
) -> str:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    llm = get_chat_model(config)
    prompt = build_qa_prompt(question=question, context=context, config=config)
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OpenAI generation failed: {exc}") from exc

    return response.content.strip()
