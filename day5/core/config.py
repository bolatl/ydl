"""Central configuration: loads .env and exposes constants used across the project."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = day5/ (this file lives in day5/core/)
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# --- Chat model API (OpenAI-compatible) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4").strip()

# --- Embedding API (separate key/model from chat on this provider) ---
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL).strip()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

# --- Email (optional, Phase 3) ---
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY", "").strip()
# MailerSend requires the "from" address to be on a verified domain. Trial
# accounts get a sandbox domain (e.g. MS_xxxx@trial-yyy.mlsender.net).
MAILERSEND_FROM_EMAIL = os.getenv("MAILERSEND_FROM_EMAIL", "").strip()
MAILERSEND_FROM_NAME = os.getenv("MAILERSEND_FROM_NAME", "Грант-помощник фонда Есенова").strip()

# --- Telegram (optional, Phase 4) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# --- Languages ---
# UI/answer languages. The corpus is Russian (parallel translations on the site);
# we retrieve in Russian and answer in the user's chosen language.
LANGUAGES = {"ru": "Русский", "kk": "Қазақша", "en": "English"}
DEFAULT_LANG = "ru"
# Instruction appended to the system prompt to set the answer language.
LANG_INSTRUCTION = {
    "ru": "Отвечай пользователю на русском языке.",
    "kk": "Пайдаланушыға қазақ тілінде жауап бер.",
    "en": "Answer the user in English.",
}
# Refusal shown when RAG has no relevant context, per language.
REFUSALS = {
    "ru": "В моих данных нет точной информации по этому вопросу.",
    "kk": "Менің деректерімде бұл сұрақ бойынша нақты ақпарат жоқ.",
    "en": "I don't have accurate information on this in my data.",
}


SOURCES_LABEL = {"ru": "Источники", "kk": "Дереккөздер", "en": "Sources"}

# Appended when the bot has no answer — invites the user to leave a request that
# will be forwarded to the foundation's team.
REQUEST_OFFER = {
    "ru": "Если хотите, я могу передать ваш вопрос команде фонда. Напишите ваше имя, "
    "email или телефон и кратко суть запроса.",
    "kk": "Қаласаңыз, сұрағыңызды қор командасына жеткізе аламын. Атыңызды, email немесе "
    "телефоныңызды және сұранысыңыздың қысқаша мәнін жазыңыз.",
    "en": "If you'd like, I can forward your question to the foundation's team. Just share "
    "your name, email or phone, and a short description of your request.",
}


def request_offer(lang: str) -> str:
    return REQUEST_OFFER.get(lang, REQUEST_OFFER[DEFAULT_LANG])


def refusal(lang: str) -> str:
    return REFUSALS.get(lang, REFUSALS[DEFAULT_LANG])


# --- Data / index paths ---
DATA_DIR = BASE_DIR / "data"
RAW_PAGES_PATH = DATA_DIR / "raw_pages.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = "yessenov_grants"
BOT_MEMORY_PATH = BASE_DIR / "bot_memory.sqlite"  # Telegram per-chat memory

# --- Scraping ---
SITE_ROOT = "https://yessenovfoundation.org"
SCRAPE_LANG_PREFIX = "/ru"  # we answer in Russian; scrape the RU pages
SCRAPE_MAX_PAGES = 100
SCRAPE_MAX_PDFS = 120  # PDFs (winner lists, regulations) are scraped in addition to pages
SCRAPE_DELAY_SEC = 0.5
MIN_PAGE_CHARS = 200  # skip near-empty pages
MIN_PDF_CHARS = 80    # PDFs can be short (e.g. a winners table); keep a low floor

# --- Chunking ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# PDFs are usually short tables (winner lists, regulations) where completeness
# matters — splitting a 20-name list across chunks means retrieval may bring back
# only half. Keep a PDF whole up to this size; only longer PDFs get chunked.
PDF_CHUNK_SIZE = 4000

# --- RAG ---
N_RESULTS = 8
# Chroma cosine distance: chunks above this are treated as irrelevant and dropped,
# so off-topic questions short-circuit to a refusal without calling the LLM.
# Tuned on text-1024: relevant content lands <=0.58, off-topic >=0.74.
MAX_DISTANCE = 0.7

# --- LLM generation ---
TEMPERATURE = 0.1

# Default (Russian) refusal — kept for backward compatibility; per-language
# refusals live in REFUSALS / refusal().
REFUSAL_TEXT = REFUSALS["ru"]


def require_llm_config() -> None:
    """Raise a clear error if the model API is not configured."""
    missing = [
        name
        for name, val in (
            ("LLM_BASE_URL", LLM_BASE_URL),
            ("LLM_API_KEY", LLM_API_KEY),
            ("LLM_MODEL", LLM_MODEL),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing model API config in .env: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in the course-issued values."
        )


def require_email_config() -> None:
    """Raise a clear error if the MailerSend email feature is not configured."""
    missing = [
        name
        for name, val in (
            ("MAILERSEND_API_KEY", MAILERSEND_API_KEY),
            ("MAILERSEND_FROM_EMAIL", MAILERSEND_FROM_EMAIL),
            ("ADMIN_EMAIL", ADMIN_EMAIL),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Email summary is not configured. Missing in .env: "
            + ", ".join(missing)
            + "."
        )
