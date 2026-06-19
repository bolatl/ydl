"""Scrape Russian-language pages from yessenovfoundation.org into data/raw_pages.jsonl.

The site is a WPML WordPress install: Kazakh is the default language (no path
prefix) and Russian lives under /ru/. We normalize every internal link to its
/ru/ form so the corpus is consistently Russian.

Usage:
    python scrape.py
"""
from __future__ import annotations

import html as html_lib
import io
import json
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

from core import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (YDL2026 educational grants bot; contact: student)"
}

# File extensions we never want to fetch as pages. (.pdf is handled separately.)
SKIP_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".mp4", ".mp3", ".css", ".js", ".woff", ".woff2", ".ttf",
)

SITE_HOST = urlparse(config.SITE_ROOT).netloc


def normalize_to_ru(url: str) -> str | None:
    """Return a same-host /ru/-prefixed URL, or None if the link should be skipped."""
    url, _ = urldefrag(url)  # drop #anchors
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc != SITE_HOST:  # same host only (skips lib.* subdomain too)
        return None
    if parsed.path.lower().endswith(SKIP_EXT) or parsed.path.lower().endswith(".pdf"):
        return None  # PDFs are collected separately, not as /ru pages

    # Strip any leading language segment (ru/kk/en/kz), then force /ru on.
    # The site nests languages (e.g. /en/, /kk/), so without this we'd pull
    # English/Kazakh pages into the Russian corpus.
    segments = [s for s in parsed.path.split("/") if s]
    if segments and segments[0].lower() in ("ru", "kk", "kz", "en"):
        segments = segments[1:]
    path = "/" + "/".join(segments)
    ru_path = "/ru" + path
    if not ru_path.endswith("/") and "." not in ru_path.rsplit("/", 1)[-1]:
        ru_path += "/"

    return f"https://{SITE_HOST}{ru_path}"


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines / trailing whitespace.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for src in (soup.find("h1"), soup.title):
        if src and src.get_text(strip=True):
            title = src.get_text(strip=True)
            # Drop the site-name suffix WordPress appends to <title>.
            return title.split("|")[0].split("—")[0].strip()
    return ""


def extract_text_from_pdf(data: bytes) -> str:
    import pypdf

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    lines = [ln.strip() for ln in "\n".join(parts).splitlines()]
    return "\n".join(ln for ln in lines if ln)


# Russian keyword hints for common PDF filename slugs. The site names files in
# transliterated Russian (e.g. "spisok-pobeditelej.pdf"), which embeds poorly
# against a Cyrillic query, so we map known slugs to a real Russian label.
_SLUG_KEYWORDS = {
    "pobeditel": "Список победителей",
    "pobedit": "Список победителей",
    "winner": "Список победителей",
    "uchastnik": "Список участников",
    "participant": "Список участников",
    "shortlist": "Список участников",
    "short-list": "Список участников",
    "polozhenie": "Положение программы: правила, требования, условия участия",
    "pravila": "Правила и требования программы",
    "programma": "Программа",
    "stipend": "Стипендия Есенова",
    "grant": "Грант",
    "expert": "Экспертный совет",
}


def pdf_label_from_url(url: str) -> str:
    """Build a Russian label for a PDF from its filename slug, e.g.
    ".../spisok-pobeditelej-1.pdf" -> "Список победителей"."""
    slug = urlparse(url).path.rsplit("/", 1)[-1].lower()
    slug = re.sub(r"\.pdf$", "", slug)
    labels = []
    for key, label in _SLUG_KEYWORDS.items():
        if key in slug and label not in labels:
            labels.append(label)
    return ". ".join(labels)


_PDF_RE = re.compile(r"https?://[^\s\"'<>()]+?\.pdf", re.IGNORECASE)


def find_pdf_urls(html: str) -> list[str]:
    """Find same-host PDF URLs in raw HTML.

    PDFs are linked through a WordPress click-counter plugin via onclick handlers
    like ...?download=1&kccpid=..&kcccount=https://.../file.pdf (not <a href>), so a
    regex over the unescaped HTML is the only reliable way to catch them.
    """
    text = html_lib.unescape(html)
    found: list[str] = []
    seen: set[str] = set()
    for match in _PDF_RE.findall(text):
        url = match.split("kcccount=")[-1]  # unwrap download-counter URLs
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        if parsed.netloc == SITE_HOST and parsed.path.lower().endswith(".pdf"):
            if url not in seen:
                seen.add(url)
                found.append(url)
    return found


def extract_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    """Return (page_links normalized to /ru, same-host pdf_links)."""
    soup = BeautifulSoup(html, "lxml")
    page_links = []
    for a in soup.find_all("a", href=True):
        absolute, _ = urldefrag(urljoin(base_url, a["href"]))
        norm = normalize_to_ru(absolute)
        if norm:
            page_links.append(norm)
    return page_links, find_pdf_urls(html)


def crawl() -> list[dict]:
    seeds = [
        config.SITE_ROOT + "/ru/",
        config.SITE_ROOT + "/ru/about-us/programs/",
        config.SITE_ROOT + "/ru/about-us/mission-and-reports/",
        config.SITE_ROOT + "/ru/contacts/",
        config.SITE_ROOT + "/ru/about-us/contacts/",
    ]
    queue: deque[str] = deque(seeds)
    seen: set[str] = set(seeds)
    pages: list[dict] = []
    pdf_urls: list[str] = []
    pdf_seen: set[str] = set()
    pdf_context: dict[str, str] = {}  # pdf_url -> title of the page that linked it

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(pages) < config.SCRAPE_MAX_PAGES:
        url = queue.popleft()
        try:
            resp = session.get(url, timeout=20, allow_redirects=True)
        except requests.RequestException as exc:
            print(f"[skip] {url} -> {exc}")
            continue

        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            print(f"[skip] {url} -> status {resp.status_code}")
            continue

        page_title = extract_title(resp.text)
        text = extract_text(resp.text)
        if len(text) >= config.MIN_PAGE_CHARS:
            pages.append({"url": url, "context": page_title, "text": text})
            print(f"[ok]   ({len(pages):>3}) {url}  ({len(text)} chars)")
        else:
            print(f"[thin] {url} ({len(text)} chars)")

        page_links, pdf_links = extract_links(resp.text, resp.url)
        for link in page_links:
            if link not in seen:
                seen.add(link)
                queue.append(link)
        for pdf in pdf_links:
            if pdf not in pdf_seen:
                pdf_seen.add(pdf)
                pdf_urls.append(pdf)
                pdf_context[pdf] = page_title  # title of the page linking this PDF

        time.sleep(config.SCRAPE_DELAY_SEC)

    pages.extend(crawl_pdfs(pdf_urls, pdf_context, session))
    return pages


def crawl_pdfs(
    pdf_urls: list[str], pdf_context: dict[str, str], session: requests.Session
) -> list[dict]:
    out: list[dict] = []
    for url in pdf_urls[: config.SCRAPE_MAX_PDFS]:
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
        except requests.RequestException as exc:
            print(f"[pdf skip] {url} -> {exc}")
            continue
        if resp.status_code != 200:
            print(f"[pdf skip] {url} -> status {resp.status_code}")
            continue

        text = extract_text_from_pdf(resp.content)
        if len(text) >= config.MIN_PDF_CHARS:
            # Context = Russian label from the filename slug + the linking page's
            # title, so a bare table (e.g. a winners list) gets a searchable anchor
            # like "Список победителей. Yessenov Data Lab 2026".
            title = pdf_context.get(url, "")
            parts = [p for p in (pdf_label_from_url(url), title) if p]
            context = ". ".join(parts)
            out.append({"url": url, "context": context, "text": text})
            print(f"[pdf]  ({len(out):>3}) {url}  ({len(text)} chars) [{context[:50]}]")
        else:
            # Likely a scanned/image PDF with no extractable text.
            print(f"[pdf empty] {url} ({len(text)} chars)")
        time.sleep(config.SCRAPE_DELAY_SEC)
    return out


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pages = crawl()

    with open(config.RAW_PAGES_PATH, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pages)} pages -> {config.RAW_PAGES_PATH}")
    if pages:
        total = sum(len(p["text"]) for p in pages)
        print(f"Total text: {total} chars (~{total // 4} tokens)")


if __name__ == "__main__":
    main()
