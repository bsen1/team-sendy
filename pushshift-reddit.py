#!/usr/bin/env python3
"""
PullPush (Pushshift mirror) → text chunks AFTER a chosen date.

Relevancy = tanh(log1p(score)) * (exp(-ln 2 * age_days / HALF_LIFE_DAYS))**FRESHNESS_ALPHA
- Bounded to [0, 1]
- Gentler recency via FRESHNESS_ALPHA (default 0.5)

Output (TSV):
  relevancy_score    text

Each post becomes:
  - Title → one chunk
  - Body  → split into paragraph chunks (blank-line delimiter)
  - All chunks from the same post share the same relevancy score

Edit SUBREDDIT, AFTER_DATE, HALF_LIFE_DAYS as needed.
"""

import math
import re
import time
import requests
from datetime import datetime, timezone

# ==== EDIT THESE ====
SUBREDDIT        = "UCSantaBarbara"
AFTER_DATE       = "2025-05-01"      # "YYYY-MM-DD" or epoch seconds string
HALF_LIFE_DAYS   = 14.0              # 14 for “current vibe”, 30 for slower topics
FRESHNESS_ALPHA  = 0.5               # <1 softens recency; try 0.35–0.75
MIN_BODY_CHARS   = 5                 # drop body chunks shorter than this (titles are kept)
# =====================

# Fixed 'today' (UTC) for age calculation
FIXED_TODAY = datetime(2025, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
FIXED_TODAY_TS = int(FIXED_TODAY.timestamp())

BASE_URL = "https://api.pullpush.io/reddit/search/submission/"
HEADERS  = {"User-Agent": "Reddit-Fetch-Relevancy/1.0 (+research)"}

def to_epoch(date_str: str) -> int:
    s = date_str.strip()
    if s.isdigit():
        return int(s)
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def get_with_retry(session: requests.Session, params: dict, max_retries: int = 6):
    backoff = 0.5
    for _ in range(max_retries):
        if "after" in params and params["after"] is not None:
            params["after"] = int(params["after"])
        if "before" in params and params["before"] is not None:
            params["before"] = int(params["before"])
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff = min(backoff * 2, 6.0)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.RequestException:
            time.sleep(backoff); backoff = min(backoff * 2, 6.0)
    return []

def sanitize_for_tsv(s: str) -> str:
    """Strip and collapse newlines/tabs to spaces for clean TSV output."""
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_paragraphs(markdown_text: str) -> list[str]:
    """Split body text into paragraphs using blank lines."""
    if not markdown_text:
        return []
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]

def relevancy(score: int, created_ts: int) -> float:
    """
    Bounded, less-recency-heavy relevancy:
      engagement = tanh(log1p(score)) ∈ (0,1)
      freshness  = exp(-ln 2 * age_days / half_life) ∈ (0,1]
      score      = engagement * freshness**FRESHNESS_ALPHA, clamped to [0,1]
    """
    s = max(0, int(score or 0))
    age_days = max(0.0, (FIXED_TODAY_TS - int(created_ts)) / 86400.0)

    engagement = math.tanh(math.log1p(s))
    freshness  = math.exp(-math.log(2) * age_days / float(HALF_LIFE_DAYS))
    # soften recency dominance
    freshness  = freshness ** float(FRESHNESS_ALPHA)

    r = engagement * freshness
    # ensure strict bounds
    if r < 0.0: r = 0.0
    if r > 1.0: r = 1.0
    return r

def fetch_all_after(subreddit: str, after_epoch: int, sleep_s: float = 0.2):
    session = requests.Session(); session.headers.update(HEADERS)
    params = {
        "subreddit": subreddit,
        "size": 250,
        "sort": "desc",
        "sort_type": "created_utc",
        "after": int(after_epoch),
    }

    before_cursor = None

    # TSV header
    print("relevancy_score\ttext")

    while True:
        if before_cursor is not None:
            params["before"] = int(before_cursor)

        data = get_with_retry(session, params)
        if not data:
            if before_cursor is not None:
                before_cursor = int(before_cursor) - 1
                data = get_with_retry(session, params)
            if not data:
                break

        for post in data:
            created_ts = post.get("created_utc")
            if not isinstance(created_ts, (int, float)):
                continue

            score = post.get("score", 0)
            rscore = relevancy(score, int(created_ts))

            # Title chunk — always kept
            title_raw = (post.get("title") or "").strip()
            if title_raw:
                title_chunk = sanitize_for_tsv(title_raw)
                if title_chunk:
                    print(f"{rscore:.6f}\t{title_chunk}")

            # Body → split into paragraph chunks, filter out super-short ones
            body_raw = post.get("selftext") or ""
            if body_raw in ("[deleted]", "[removed]"):
                body_raw = ""
            paragraphs = split_paragraphs(body_raw)

            for para in paragraphs:
                chunk = sanitize_for_tsv(para)
                if len(chunk) < MIN_BODY_CHARS:
                    continue  # drop very short body chunks
                print(f"{rscore:.6f}\t{chunk}")

        last_created = data[-1].get("created_utc")
        if not isinstance(last_created, (int, float)):
            break
        before_cursor = int(last_created) - 1
        time.sleep(sleep_s)

if __name__ == "__main__":
    after_epoch = to_epoch(AFTER_DATE)
    fetch_all_after(SUBREDDIT, after_epoch)
