"""gemma4 client + answer generation.

`call_llm` is the single place the model is called; everything else builds prompts
on top of it. Swap only this function if the course uses a different interface.
"""
from __future__ import annotations

import json
from functools import lru_cache

from core import config

_SYSTEM_PROMPT_TMPL = """Ты — ИИ-помощник по грантам и программам фонда Шахмардана Есенова.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного RAG-контекста.
2. Не выдумывай дедлайны, суммы, требования, документы, критерии отбора и условия. \
Если такого факта нет в контексте — не пиши его.
3. Если в контексте нет точного ответа, ответь ровно: "{refusal}"
4. Если вопрос не относится к фонду, грантам, программам или стипендиям — вежливо \
скажи, что отвечаешь только по этим темам.
5. {lang_instruction} Пиши понятно, кратко и дружелюбно.
6. Не придумывай ссылки. Источники добавит система отдельно.
7. Контакты фонда (email, телефон) сообщай ТОЛЬКО если они есть в RAG-контексте; \
не выдумывай адреса и номера.
8. Если пользователь хочет подать заявку, получить помощь или связаться с фондом — ИЛИ \
если ты не можешь ответить по данным — предложи оставить заявку: попроси имя, email или \
телефон и кратко суть запроса. Скажи, что передашь это команде фонда вместе с историей \
разговора. Сам контактные данные не придумывай."""


def build_system_prompt(lang: str = config.DEFAULT_LANG) -> str:
    """System prompt with the refusal text and answer language for `lang`."""
    return _SYSTEM_PROMPT_TMPL.format(
        refusal=config.refusal(lang),
        lang_instruction=config.LANG_INSTRUCTION.get(
            lang, config.LANG_INSTRUCTION[config.DEFAULT_LANG]
        ),
    )


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    config.require_llm_config()
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL or None)


def call_llm(messages: list[dict], temperature: float = config.TEMPERATURE) -> str:
    """Single point where gemma4 is called."""
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


_REWRITE_PROMPT = """Ты помогаешь поисковой системе по сайту фонда Шахмардана Есенова.

Перепиши вопрос пользователя в один самостоятельный поисковый запрос на русском языке.
- Раскрывай разговорные сокращения и названия программ в полную форму
  (например: «дата лаба», «дата лабы 26» → «Yessenov Data Lab 2026»;
  «стипендия» → «Стипендия Есенова»).
- Используй недавний диалог только чтобы раскрыть ссылки вроде «эта программа», «он», «там».
- Сохрани и латинские, и русские варианты названия программы, если они есть.
- Не отвечай на вопрос и не добавляй фактов, которых нет. Верни только запрос.

Недавний диалог:
{history}

Вопрос пользователя: {question}

Поисковый запрос:"""


def rewrite_question_for_search(question: str, history: list[dict] | None = None) -> str:
    """Turn a colloquial/follow-up question into a standalone search query.

    Improves retrieval for phrasings like "дата лабы 26 победители". Falls back to
    the original question on any failure, so search never breaks.
    """
    try:
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-4:])
        query = call_llm(
            [{"role": "user", "content": _REWRITE_PROMPT.format(history=hist or "—", question=question)}],
            temperature=0.0,
        )
        query = query.strip().strip('"').splitlines()[0] if query.strip() else ""
        return query or question
    except Exception:
        return question


_SUMMARY_PROMPT = """Ты составляешь краткое саммари диалога пользователя с ботом-помощником \
по грантам фонда Шахмардана Есенова — для отправки администратору по email.

Сделай сжатое, деловое саммари на русском языке:
- какие темы/программы/гранты обсуждались;
- какие вопросы задавал пользователь;
- какие вопросы остались без точного ответа (бот ответил, что данных нет).

Не выдумывай факты, которых не было в диалоге. Пиши кратко, по пунктам.

Диалог:
{conversation}

Саммари:"""


def summarize_chat(messages: list[dict]) -> str:
    """Summarize a [{role, content}] conversation into a short admin-facing digest."""
    conversation = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
        for m in messages
    )
    return call_llm(
        [{"role": "user", "content": _SUMMARY_PROMPT.format(conversation=conversation)}],
        temperature=0.2,
    )


_MEMORY_PROMPT = """Ты обновляешь краткую память диалога Telegram-бота по грантам фонда \
Шахмардана Есенова, чтобы бот понимал дальнейшие follow-up вопросы.

Сохрани: какую программу/грант/тему обсуждает пользователь; параметры, которые он сам \
сообщил; уже заданные и нерешённые вопросы; объекты ссылок («эта программа», «этот грант»).
Не выдумывай факты и не превращай ответы бота в источник истины. Пиши кратко (до 1000 \
символов).

Старое саммари:
{old_summary}

Сообщения для сжатия:
{messages}

Новое саммари:"""


def summarize_memory(old_summary: str, messages: list[dict]) -> str:
    """Fold older turns into a rolling summary for long Telegram conversations."""
    msgs = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    try:
        return call_llm(
            [
                {
                    "role": "user",
                    "content": _MEMORY_PROMPT.format(
                        old_summary=old_summary or "—", messages=msgs
                    ),
                }
            ],
            temperature=0.2,
        )
    except Exception:
        return old_summary


_EXTRACT_PROMPT = """Проанализируй диалог пользователя с ботом фонда Шахмардана Есенова.

Определи, оставил ли пользователь заявку/обращение, которое нужно передать команде фонда.
Заявка считается готовой к отправке (ready=true) ТОЛЬКО если в диалоге есть И контакт
пользователя (email или телефон), И суть запроса.

Верни СТРОГО JSON без пояснений, в формате:
{{"ready": true|false, "name": "...", "contact": "...", "request": "..."}}

- name — имя пользователя, если он его назвал, иначе "".
- contact — email или телефон, который указал пользователь, иначе "".
- request — кратко суть того, что хочет пользователь, иначе "".
- Не выдумывай данные, которых нет в диалоге.

Диалог:
{conversation}

JSON:"""


def extract_contact_request(messages: list[dict]) -> dict | None:
    """Detect whether the user has left a complete contact request to forward.

    Returns {"name", "contact", "request"} if ready, else None. Never raises —
    extraction failures just mean "no request detected".
    """
    if not messages:
        return None
    conversation = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
        for m in messages
    )
    try:
        raw = call_llm(
            [{"role": "user", "content": _EXTRACT_PROMPT.format(conversation=conversation)}],
            temperature=0.0,
        )
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        if data.get("ready") and data.get("contact") and data.get("request"):
            return {
                "name": str(data.get("name", "")).strip(),
                "contact": str(data["contact"]).strip(),
                "request": str(data["request"]).strip(),
            }
    except Exception:
        return None
    return None


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Фрагмент {i}] (источник: {c.get('url','')})\n{c['text']}")
    return "\n\n".join(blocks)


def _unique_sources(chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        url = c.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def ask_gemma(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    lang: str = config.DEFAULT_LANG,
) -> str:
    """Generate an answer grounded in `chunks`, in language `lang`.

    Anti-hallucination: if there are no chunks, refuse WITHOUT calling the model.
    `history` (optional) is prior [{role, content}] turns for conversational context;
    it never serves as a source of grant facts.
    """
    refusal = config.refusal(lang)
    if not chunks:
        # No relevant data — refuse honestly, but offer to forward a request.
        return f"{refusal}\n\n{config.request_offer(lang)}"

    messages = [{"role": "system", "content": build_system_prompt(lang)}]
    if history:
        messages.append(
            {
                "role": "system",
                "content": "Недавний диалог (только для понимания контекста вопроса, "
                "НЕ источник фактов):\n"
                + "\n".join(f"{m['role']}: {m['content']}" for m in history),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"RAG-контекст:\n{_format_context(chunks)}\n\n"
                f"Вопрос пользователя: {question}\n\n"
                "Ответь по правилам, опираясь только на контекст."
            ),
        }
    )

    answer = call_llm(messages)

    sources = _unique_sources(chunks)
    if sources and answer != refusal:
        label = config.SOURCES_LABEL.get(lang, config.SOURCES_LABEL[config.DEFAULT_LANG])
        answer += f"\n\n{label}:\n" + "\n".join(f"• {u}" for u in sources)
    return answer
