#!/usr/bin/env python3
"""
Stocktwits ticker scraper → CSV/JSONL with NLP-ready segment extraction.

Examples:
  python stocktwits_scrape.py --ticker TSLA --limit 1500 --out tsla.csv
  python stocktwits_scrape.py -t AAPL -o aapl.jsonl --max-segments 8 --min-words 3 -v
"""

import argparse
import csv
import json
import sys
import time
import re
from typing import Dict, Any, List, Optional, Tuple, Pattern

import requests

from sp500_ticker_map import TICKER_TO_NAME  # {ticker: (cleaned_name, [tokens])}

API_TEMPLATE = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
HEADERS = {
    # A descriptive UA helps platform operators contact you if needed
    "User-Agent": "Academic-Research-SentimentBot/1.2 (+contact: your_email@example.com)"
}

SentencePattern = Pattern[str]

# -------------------------- mention building ---------------------------------

def _escape_for_regex(s: str) -> str:
    return re.escape(s)

def build_mention_regex(ticker: str) -> Tuple[SentencePattern, List[str]]:
    """
    Build a case-insensitive regex that hits on:
      - Cashtag: $AAPL
      - Ticker:  AAPL (word boundary)
      - Cleaned company full name phrase (if available)
      - Any token from the cleaned company name (word boundary)
    Returns (compiled_pattern, debug_terms).
    """
    tkr = ticker.upper()
    terms: List[str] = []

    # Always include ticker and cashtag
    terms.append(rf"\${_escape_for_regex(tkr)}\b")
    terms.append(rf"\b{_escape_for_regex(tkr)}\b")

    cname = None
    tokens: List[str] = []
    info = TICKER_TO_NAME.get(tkr)
    if isinstance(info, tuple) and len(info) == 2:
        cname, tokens = info

    # Full company phrase
    if cname:
        cname_norm = re.sub(r"\s+", " ", cname).strip()
        if cname_norm:
            terms.append(_escape_for_regex(cname_norm))

    # Individual tokens (drop super-short tokens)
    for tok in (tokens or []):
        tok_norm = tok.strip()
        if len(tok_norm) >= 2:
            terms.append(rf"\b{_escape_for_regex(tok_norm)}\b")

    # Dedup while preserving order
    seen = set()
    unique_terms: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)

    if not unique_terms:
        unique_terms = [rf"\${_escape_for_regex(tkr)}\b", rf"\b{_escape_for_regex(tkr)}\b"]

    pattern_str = "(?i)" + "(" + "|".join(unique_terms) + ")"
    try:
        pattern = re.compile(pattern_str)
    except re.error:
        pattern = re.compile(rf"(?i)\b{_escape_for_regex(tkr)}\b")

    return pattern, unique_terms


# ----------------------------- NLP extraction --------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_URL_RE = re.compile(r"https?://\S+")
_CASHTAG_RE = re.compile(r"\$\w+")
_HANDLE_RE = re.compile(r"[@#]\w+")
# Strip most non-alphanumerics except common punctuation useful for sentiment
_NOISE_RE = re.compile(r"[^A-Za-z0-9.,!?'\s]+")

def clean_fragment(s: str) -> str:
    """Aggressively clean a text fragment for NLP sentiment analysis."""
    s = _URL_RE.sub("", s)
    s = _CASHTAG_RE.sub("", s)
    s = _HANDLE_RE.sub("", s)
    s = _NOISE_RE.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def extract_key_segments(
    text: str,
    mention_re: SentencePattern,
    max_segments: int = 5,
    min_words: int = 4,
) -> List[str]:
    """
    Extract short, relevant segments suited for sentiment analysis:
      - Keep sentences that mention ticker/company/tokens.
      - Clean aggressively (URLs, cashtags, handles, symbols).
      - Filter by min word count.
      - Merge adjacent short matched sentences (for brief multi-sentence opinions).
    """
    if not text:
        return []

    # Normalize line breaks, split into sentences
    norm = text.replace("\r\n", "\n").replace("\n", " ").strip()
    sentences = _SENT_SPLIT.split(norm) if norm else []

    hits_raw: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if mention_re.search(s):
            s_clean = clean_fragment(s)
            if len(s_clean.split()) >= min_words:
                hits_raw.append(s_clean)

    # Merge adjacent short ones (< 40 chars each)
    merged: List[str] = []
    i = 0
    while i < len(hits_raw):
        cur = hits_raw[i]
        if i + 1 < len(hits_raw) and len(cur) < 40 and len(hits_raw[i + 1]) < 40:
            merged.append(f"{cur} {hits_raw[i + 1]}")
            i += 2
        else:
            merged.append(cur)
            i += 1

    # Dedup, cap
    seen = set()
    final: List[str] = []
    for s in merged:
        if s not in seen:
            seen.add(s)
            final.append(s)
            if len(final) >= max_segments:
                break
    return final


# ----------------------------- HTTP / parsing --------------------------------

def fetch_page(ticker: str, max_id: Optional[int] = None) -> Dict[str, Any]:
    url = API_TEMPLATE.format(ticker=ticker.upper())
    params = {}
    if max_id is not None:
        params["max"] = max_id  # returns messages with id <= max
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()

def normalize_message(
    m: Dict[str, Any],
    ticker: str,
    mention_re: SentencePattern,
    max_segments: int,
    min_words: int,
    lowercase_nlp: bool,
) -> Dict[str, Any]:
    """
    Return compact payload with NLP-ready fields:
      - ticker
      - body (collapsed)
      - key_segments: list[str] (cleaned, short)
      - nlp_text: " " joined segments (optional lowercased)
    """
    body = (m.get("body") or "").replace("\r\n", "\n").replace("\n", " ").strip()
    key_segments = extract_key_segments(
        body, mention_re, max_segments=max_segments, min_words=min_words
    )
    nlp_text = " ".join(key_segments)
    if lowercase_nlp:
        nlp_text = nlp_text.lower()
    return {
        "ticker": ticker.upper(),
        "body": body,
        "key_segments": key_segments,
        "nlp_text": nlp_text,
    }


# ----------------------------- writers ---------------------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["ticker", "body", "key_segments", "nlp_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r2 = r.copy()
            r2["key_segments"] = " | ".join(r2.get("key_segments") or [])
            r2["body"] = (r2.get("body") or "").replace("\n", " ").strip()
            writer.writerow(r2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------- scraper ---------------------------------------

def scrape(
    ticker: str,
    limit: int,
    sleep_s: float,
    verbose: bool,
    mention_re: SentencePattern,
    max_segments: int,
    min_words: int,
    lowercase_nlp: bool,
    drop_empty: bool,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    max_id: Optional[int] = None
    seen_ids = set()

    while len(collected) < limit:
        try:
            data = fetch_page(ticker, max_id=max_id)
        except Exception as e:
            print(f"[!] Error: {e}", file=sys.stderr)
            break

        cursor = data.get("cursor") or (data.get("response", {}) or {}).get("cursor") or {}
        messages = data.get("messages") or []

        if verbose:
            print(f"[i] Fetched {len(messages)} msgs (max_id={max_id})")

        if not messages:
            break

        for m in messages:
            mid = m.get("id")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            row = normalize_message(
                m,
                ticker,
                mention_re,
                max_segments=max_segments,
                min_words=min_words,
                lowercase_nlp=lowercase_nlp,
            )

            if drop_empty and not row["key_segments"]:
                continue

            collected.append(row)
            if len(collected) >= limit:
                break

        # Paginate backward using cursor or the smallest id we saw
        next_max = None
        if isinstance(cursor, dict) and cursor.get("max"):
            next_max = cursor["max"] - 1
        else:
            smallest = min((msg.get("id") for msg in messages if isinstance(msg.get("id"), int)), default=None)
            if smallest is not None:
                next_max = smallest - 1

        if next_max is None or next_max <= 0:
            break

        max_id = next_max
        time.sleep(sleep_s)  # be polite

    return collected


# ----------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Scrape Stocktwits messages for a ticker, extracting NLP-ready key segments mentioning the ticker/company."
    )
    ap.add_argument("-t", "--ticker", required=True, help="Ticker symbol, e.g., TSLA")
    ap.add_argument("-l", "--limit", type=int, default=1000, help="Max messages to keep (default 1000)")
    ap.add_argument("-o", "--out", required=True, help="Output file path (.csv or .jsonl)")
    ap.add_argument("--sleep", type=float, default=0.6, help="Seconds to sleep between requests (default 0.6)")
    ap.add_argument("--max-segments", type=int, default=5, help="Max key segments per message (default 5)")
    ap.add_argument("--min-words", type=int, default=4, help="Minimum words per kept segment (default 4)")
    ap.add_argument("--lower", action="store_true", help="Lowercase nlp_text")
    ap.add_argument("--drop-empty", action="store_true", help="Drop messages with no extracted segments")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print progress")
    args = ap.parse_args()

    # Build mention regex from S&P 500 map (ticker, company name, tokens)
    mention_re, terms = build_mention_regex(args.ticker)
    if args.verbose:
        preview = ", ".join(terms[:10])
        print(f"[i] Using mention terms for {args.ticker}: {preview}{' ...' if len(terms) > 10 else ''}")

    rows = scrape(
        args.ticker,
        args.limit,
        args.sleep,
        args.verbose,
        mention_re=mention_re,
        max_segments=args.max_segments,
        min_words=args.min_words,
        lowercase_nlp=args.lower,
        drop_empty=args.drop_empty,
    )

    if args.out.lower().endswith(".csv"):
        write_csv(args.out, rows)
    elif args.out.lower().endswith(".jsonl"):
        write_jsonl(args.out, rows)
    else:
        print("[!] Please use .csv or .jsonl for the output file extension.", file=sys.stderr)
        sys.exit(1)

    print(f"[✓] Saved {len(rows)} messages to {args.out}")


if __name__ == "__main__":
    main()
