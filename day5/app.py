"""Streamlit chat UI for the Yessenov grants RAG bot.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import subprocess
import sys

import streamlit as st

from core import config
from core.llm import (
    ask_gemma,
    extract_contact_request,
    rewrite_question_for_search,
    summarize_chat,
)
from core.mailer import send_contact_request, send_summary_email
from core.rag import search_context

EMAIL_ENABLED = bool(config.MAILERSEND_API_KEY and config.ADMIN_EMAIL)

# How many prior turns to pass as conversational context (clarity only, not facts).
HISTORY_TURNS = 6

# Localized UI strings. Bot *answers* are localized via core.llm; this is the chrome.
STRINGS = {
    "ru": {
        "page_title": "Грант-помощник · фонд Есенова",
        "title": "🎓 Грант-помощник фонда Есенова",
        "lang_label": "Язык",
        "about": "О боте",
        "caption": "RAG-бот на gemma4. Факты берутся только из найденных фрагментов сайта "
        "yessenovfoundation.org. Память диалога помогает понять вопрос, но не служит "
        "источником фактов.",
        "clear": "🗑 Очистить чат",
        "welcome": "Привет! Я помощник по грантам и программам фонда Шахмардана Есенова.\n\n"
        "Спросите, например:\n• Какие программы есть у фонда?\n• Кто может подать заявку?\n"
        "• Какие документы нужны?\n\nЯ отвечаю только по данным с сайта фонда. Если "
        "информации нет — я честно скажу, что не знаю, и не буду выдумывать.",
        "input": "Ваш вопрос о грантах и программах...",
        "spinner": "Ищу в данных фонда...",
        "error": "⚠️ Ошибка при обработке запроса. Проверьте, что выполнены `python scrape.py` "
        "и `python build_index.py`, а в `.env` указан доступ к модели.\n\n`{exc}`",
        "email_btn": "✉️ Отправить саммари администратору",
        "email_empty": "Диалог пуст — нечего отправлять.",
        "email_spinner": "Готовлю и отправляю саммари...",
        "email_ok": "Саммари отправлено на {email}.",
        "email_fail": "Не удалось отправить письмо: {exc}",
        "lead_title": "**📨 Похоже, вы хотите оставить заявку. Передать её в фонд?**",
        "lead_name": "Имя",
        "lead_contact": "Контакт",
        "lead_request": "Запрос",
        "lead_send": "✅ Отправить заявку",
        "lead_cancel": "Отмена",
        "lead_ok": "Заявка передана в фонд (на {email}).",
        "lead_fail": "Не удалось отправить заявку: {exc}",
        "admin": "Администрирование",
        "db_btn": "🔄 Обновить базу данных",
        "db_hint": "Заново скачает страницы сайта и пересоберёт индекс. Занимает несколько минут.",
        "db_running": "Обновляю базу: скрапинг и пересборка индекса... это займёт несколько минут.",
        "db_ok": "База данных обновлена.",
        "db_fail": "Не удалось обновить базу: {exc}",
    },
    "kk": {
        "page_title": "Грант-көмекші · Есенов қоры",
        "title": "🎓 Есенов қорының грант-көмекшісі",
        "lang_label": "Тіл",
        "about": "Бот туралы",
        "caption": "gemma4 негізіндегі RAG-бот. Деректер тек yessenovfoundation.org сайтының "
        "табылған үзінділерінен алынады. Диалог жады сұрақты түсінуге көмектеседі, бірақ "
        "дерек көзі емес.",
        "clear": "🗑 Чатты тазалау",
        "welcome": "Сәлем! Мен Шахмардан Есенов қорының гранттары мен бағдарламалары бойынша "
        "көмекшімін.\n\nМысалы, сұраңыз:\n• Қорда қандай бағдарламалар бар?\n• Кім өтінім бере "
        "алады?\n• Қандай құжаттар қажет?\n\nМен тек қор сайтының деректері бойынша жауап "
        "беремін. Ақпарат болмаса — ойдан шығармай, білмейтінімді адал айтамын.",
        "input": "Гранттар мен бағдарламалар туралы сұрағыңыз...",
        "spinner": "Қор деректерінен іздеудемін...",
        "error": "⚠️ Сұранысты өңдеу қатесі. `python scrape.py` және `python build_index.py` "
        "орындалғанын, `.env`-те модель қолжетімділігі бар екенін тексеріңіз.\n\n`{exc}`",
        "email_btn": "✉️ Әкімшіге қысқаша есеп жіберу",
        "email_empty": "Диалог бос — жіберетін ештеңе жоқ.",
        "email_spinner": "Қысқаша есеп дайындалып жіберілуде...",
        "email_ok": "Қысқаша есеп {email} мекенжайына жіберілді.",
        "email_fail": "Хат жіберілмеді: {exc}",
        "lead_title": "**📨 Сіз өтінім қалдырғыңыз келетін сияқты. Оны қорға жіберейік пе?**",
        "lead_name": "Аты",
        "lead_contact": "Байланыс",
        "lead_request": "Сұраныс",
        "lead_send": "✅ Өтінімді жіберу",
        "lead_cancel": "Бас тарту",
        "lead_ok": "Өтінім қорға жіберілді ({email}).",
        "lead_fail": "Өтінім жіберілмеді: {exc}",
        "admin": "Әкімшілік",
        "db_btn": "🔄 Дерекқорды жаңарту",
        "db_hint": "Сайт беттерін қайта жүктеп, индексті қайта жинайды. Бірнеше минут алады.",
        "db_running": "Дерекқор жаңартылуда: скрапинг және индексті қайта жинау... бірнеше минут.",
        "db_ok": "Дерекқор жаңартылды.",
        "db_fail": "Дерекқорды жаңарту мүмкін болмады: {exc}",
    },
    "en": {
        "page_title": "Grants assistant · Yessenov Foundation",
        "title": "🎓 Yessenov Foundation grants assistant",
        "lang_label": "Language",
        "about": "About",
        "caption": "RAG bot on gemma4. Facts come only from retrieved fragments of "
        "yessenovfoundation.org. Chat memory helps interpret the question but is not a "
        "source of facts.",
        "clear": "🗑 Clear chat",
        "welcome": "Hi! I'm an assistant for the grants and programs of the Shakhmardan "
        "Yessenov Foundation.\n\nTry asking:\n• What programs does the foundation have?\n"
        "• Who can apply?\n• What documents are needed?\n\nI answer only from the "
        "foundation's website data. If I don't have the information, I'll honestly say so "
        "rather than make it up.",
        "input": "Your question about grants and programs...",
        "spinner": "Searching the foundation's data...",
        "error": "⚠️ Error while processing the request. Make sure `python scrape.py` and "
        "`python build_index.py` have run and the model access is set in `.env`.\n\n`{exc}`",
        "email_btn": "✉️ Email summary to admin",
        "email_empty": "The conversation is empty — nothing to send.",
        "email_spinner": "Preparing and sending the summary...",
        "email_ok": "Summary sent to {email}.",
        "email_fail": "Could not send the email: {exc}",
        "lead_title": "**📨 Looks like you'd like to leave a request. Forward it to the foundation?**",
        "lead_name": "Name",
        "lead_contact": "Contact",
        "lead_request": "Request",
        "lead_send": "✅ Send request",
        "lead_cancel": "Cancel",
        "lead_ok": "Request forwarded to the foundation ({email}).",
        "lead_fail": "Could not send the request: {exc}",
        "admin": "Administration",
        "db_btn": "🔄 Refresh database",
        "db_hint": "Re-scrapes the site pages and rebuilds the index. Takes a few minutes.",
        "db_running": "Refreshing database: scraping and rebuilding the index... a few minutes.",
        "db_ok": "Database refreshed.",
        "db_fail": "Could not refresh the database: {exc}",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = config.DEFAULT_LANG
if "messages" not in st.session_state:
    st.session_state.messages = []

lang = st.session_state.lang
T = STRINGS[lang]

st.set_page_config(page_title=T["page_title"], page_icon="🎓")


def _refresh_database() -> None:
    """Re-run scrape.py then build_index.py as subprocesses, then drop the cached
    Chroma handle so the next query uses the freshly built index."""
    for script in ("scrape.py", "build_index.py"):
        proc = subprocess.run(
            [sys.executable, script],
            cwd=str(config.BASE_DIR),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError(f"{script} failed:\n{tail}")
    from core import rag

    rag._collection.cache_clear()


with st.sidebar:
    # Language selector — switching re-renders the chrome; chat history is kept.
    codes = list(config.LANGUAGES)
    st.session_state.lang = st.radio(
        T["lang_label"],
        codes,
        index=codes.index(lang),
        format_func=lambda c: config.LANGUAGES[c],
        horizontal=True,
    )
    if st.session_state.lang != lang:
        st.rerun()

    st.header(T["about"])
    st.caption(T["caption"])
    if st.button(T["clear"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_request", None)
        st.rerun()

    # Phase 3 (optional): email a summary of this chat to the admin, on explicit click.
    if EMAIL_ENABLED:
        st.divider()
        if st.button(T["email_btn"], use_container_width=True):
            if not st.session_state.messages:
                st.warning(T["email_empty"])
            else:
                with st.spinner(T["email_spinner"]):
                    try:
                        summary = summarize_chat(st.session_state.messages)
                        send_summary_email(summary)
                        st.success(T["email_ok"].format(email=config.ADMIN_EMAIL))
                    except Exception as exc:
                        st.error(T["email_fail"].format(exc=exc))

    # Admin: re-scrape the site and rebuild the index.
    st.divider()
    with st.expander(T["admin"]):
        st.caption(T["db_hint"])
        if st.button(T["db_btn"], use_container_width=True):
            with st.spinner(T["db_running"]):
                try:
                    _refresh_database()
                    st.success(T["db_ok"])
                except Exception as exc:
                    st.error(T["db_fail"].format(exc=exc))

st.title(T["title"])

if not st.session_state.messages:
    st.info(T["welcome"])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _history_for_llm() -> list[dict]:
    """Recent turns, excluding the just-added user question (added below)."""
    return st.session_state.messages[-(HISTORY_TURNS * 2):]


if prompt := st.chat_input(T["input"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(T["spinner"]):
            try:
                history = _history_for_llm()[:-1]  # drop the current question
                search_query = rewrite_question_for_search(prompt, history)
                chunks = search_context(search_query)
                answer = ask_gemma(prompt, chunks, history=history, lang=lang)
            except Exception as exc:
                answer = T["error"].format(exc=exc)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Lead capture: if the user has left a complete request, surface it for sending.
    if EMAIL_ENABLED:
        req = extract_contact_request(st.session_state.messages)
        if req:
            st.session_state.pending_request = req


def _render_pending_request() -> None:
    """Confirmation box for a detected user request. Sends only on explicit click."""
    req = st.session_state.get("pending_request")
    if not req:
        return
    with st.container(border=True):
        st.markdown(T["lead_title"])
        st.markdown(
            f"- **{T['lead_name']}:** {req['name'] or '—'}\n"
            f"- **{T['lead_contact']}:** {req['contact']}\n"
            f"- **{T['lead_request']}:** {req['request']}"
        )
        col_send, col_cancel = st.columns(2)
        if col_send.button(T["lead_send"], use_container_width=True):
            try:
                send_contact_request(
                    req["name"],
                    req["contact"],
                    req["request"],
                    conversation=st.session_state.messages,
                )
                st.session_state.pop("pending_request", None)
                st.success(T["lead_ok"].format(email=config.ADMIN_EMAIL))
            except Exception as exc:
                st.error(T["lead_fail"].format(exc=exc))
        if col_cancel.button(T["lead_cancel"], use_container_width=True):
            st.session_state.pop("pending_request", None)
            st.rerun()


if EMAIL_ENABLED:
    _render_pending_request()
