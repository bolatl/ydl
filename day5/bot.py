"""Telegram front-end for the Yessenov grants RAG bot.

Reuses the same UI-agnostic core/ as the Streamlit app: retrieval, the gemma4
client and anti-hallucination rules are identical. Adds per-chat SQLite memory,
language switching (RU/KK/EN), and /reset.

Run:
    python bot.py
"""
from __future__ import annotations

import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core import config, memory
from core.llm import ask_gemma, rewrite_question_for_search, summarize_chat
from core.rag import search_context

try:
    from core.mailer import send_summary_email
except Exception:  # pragma: no cover - optional dependency path
    send_summary_email = None

EMAIL_ENABLED = bool(config.MAILERSEND_API_KEY and config.ADMIN_EMAIL and send_summary_email)
TG_LIMIT = 4000  # Telegram hard limit is 4096; keep margin.

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("yessenov-bot")

BOT_TEXT = {
    "ru": {
        "start": "Привет! Я помощник по грантам и программам фонда Шахмардана Есенова.\n\n"
        "Спросите, например: «Какие программы есть?», «Кто может подать заявку?».\n"
        "Я отвечаю только по данным с сайта фонда и не выдумываю.\n\n"
        "Команды: /lang — язык, /reset — очистить контекст, /help — помощь.",
        "choose_lang": "Выберите язык:",
        "lang_set": "Готово. Отвечаю на русском.",
        "reset": "Готово, я очистил контекст этого чата. Можем начать заново.",
        "error": "⚠️ Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
        "summary_empty": "Диалог пуст — нечего отправлять.",
        "summary_ok": "Саммари отправлено администратору.",
        "summary_fail": "Не удалось отправить саммари.",
    },
    "kk": {
        "start": "Сәлем! Мен Шахмардан Есенов қорының гранттары бойынша көмекшімін.\n\n"
        "Мысалы сұраңыз: «Қандай бағдарламалар бар?», «Кім өтінім бере алады?».\n"
        "Мен тек қор сайтының деректері бойынша жауап беремін, ойдан шығармаймын.\n\n"
        "Командалар: /lang — тіл, /reset — контекстті тазалау, /help — көмек.",
        "choose_lang": "Тілді таңдаңыз:",
        "lang_set": "Дайын. Қазақ тілінде жауап беремін.",
        "reset": "Дайын, осы чаттың контекстін тазаладым. Қайта бастай аламыз.",
        "error": "⚠️ Сұранысты өңдеу кезінде қате шықты. Қайталап көріңіз.",
        "summary_empty": "Диалог бос — жіберетін ештеңе жоқ.",
        "summary_ok": "Қысқаша есеп әкімшіге жіберілді.",
        "summary_fail": "Қысқаша есеп жіберілмеді.",
    },
    "en": {
        "start": "Hi! I'm an assistant for the Shakhmardan Yessenov Foundation's grants.\n\n"
        "Try: \"What programs are there?\", \"Who can apply?\".\n"
        "I answer only from the foundation's website data and don't make things up.\n\n"
        "Commands: /lang — language, /reset — clear context, /help — help.",
        "choose_lang": "Choose a language:",
        "lang_set": "Done. I'll answer in English.",
        "reset": "Done, I cleared this chat's context. We can start over.",
        "error": "⚠️ Something went wrong while processing the request. Please try again.",
        "summary_empty": "The conversation is empty — nothing to send.",
        "summary_ok": "Summary sent to the admin.",
        "summary_fail": "Could not send the summary.",
    },
}


def _t(lang: str, key: str) -> str:
    return BOT_TEXT.get(lang, BOT_TEXT[config.DEFAULT_LANG])[key]


def _lang_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(name, callback_data=f"lang:{code}")
          for code, name in config.LANGUAGES.items()]]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = memory.load(update.effective_chat.id)["lang"]
    await update.message.reply_text(_t(lang, "start"), reply_markup=_lang_keyboard())


async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = memory.load(update.effective_chat.id)["lang"]
    await update.message.reply_text(_t(lang, "choose_lang"), reply_markup=_lang_keyboard())


async def on_lang_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    code = query.data.split(":", 1)[1]
    if code in config.LANGUAGES:
        memory.set_lang(query.message.chat.id, code)
        await query.edit_message_text(_t(code, "lang_set"))


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = memory.load(update.effective_chat.id)["lang"]
    memory.reset(update.effective_chat.id)
    await update.message.reply_text(_t(lang, "reset"))


async def send_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = memory.load(chat_id)
    lang = state["lang"]
    if not state["recent"]:
        await update.message.reply_text(_t(lang, "summary_empty"))
        return
    try:
        summary = await asyncio.to_thread(summarize_chat, state["recent"])
        await asyncio.to_thread(send_summary_email, summary)
        await update.message.reply_text(_t(lang, "summary_ok"))
    except Exception:
        log.exception("send_summary failed")
        await update.message.reply_text(_t(lang, "summary_fail"))


def _answer(question: str, state: dict) -> str:
    """Blocking RAG pipeline (run off the event loop via asyncio.to_thread)."""
    history = memory.history_for_prompt(state)
    search_query = rewrite_question_for_search(question, history)
    chunks = search_context(search_query)
    return ask_gemma(question, chunks, history=history, lang=state["lang"])


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    question = (update.message.text or "").strip()
    if not question:
        return
    state = memory.load(chat_id)
    log.info("chat %s (%s): %s", chat_id, state["lang"], question)

    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        answer = await asyncio.to_thread(_answer, question, state)
    except Exception:
        log.exception("pipeline failed")
        await update.message.reply_text(_t(state["lang"], "error"))
        return

    await _send_long(update, answer)
    memory.add_turn(chat_id, question, answer)


async def _send_long(update: Update, text: str) -> None:
    """Split replies over Telegram's per-message limit."""
    for i in range(0, len(text), TG_LIMIT):
        await update.message.reply_text(text[i : i + TG_LIMIT], disable_web_page_preview=True)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    if EMAIL_ENABLED:
        app.add_handler(CommandHandler("send_summary", send_summary_cmd))
    app.add_handler(CallbackQueryHandler(on_lang_choice, pattern=r"^lang:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("Bot started. Email summary: %s", "on" if EMAIL_ENABLED else "off")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
