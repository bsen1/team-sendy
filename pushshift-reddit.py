#!/usr/bin/env python3
"""
PullPush (Pushshift mirror) → print posts AFTER a chosen date.

Output columns (TSV):
  score    date_posted_utc    title    body
"""

import time
import requests
from datetime import datetime, timezone

# ==== EDIT THESE ====
SUBREDDIT  = "UCLA"
AFTER_DATE = "2025-01-01"
# =====================

BASE_URL = "https://api.pullpush.io/reddit/search/submission/"
HEADERS  = {"User-Agent": "Reddit-Fetch-Robust/1.3 (+research)"}

def to_epoch(date_str: str) -> int:
    s = date_str.strip()
    if s.isdigit():
        return int(s)
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def get_with_retry(session: requests.Session, params: dict, max_retries: int = 6):
    backoff = 0.5
    for _ in range(max_retries):
        if "after" in params:  params["after"] = int(params["after"])
        if "before" in params: params["before"] = int(params["before"])
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

def sanitize_field(s: str) -> str:
    if not s: return ""
    return s.replace("\n", " ").replace("\t", " ").strip()

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
    print("score\tdate_posted_utc\ttitle\tbody")

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
            created_dt  = datetime.fromtimestamp(int(created_ts), tz=timezone.utc)
            created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S")

            score = post.get("score", 0)
            title = sanitize_field(post.get("title") or "")
            body  = sanitize_field(post.get("selftext") or "")
            if body in ("[deleted]", "[removed]"):
                body = ""

            print(f"{score}\t{created_str}\t{title}\t{body}")

        last_created = data[-1].get("created_utc")
        if not isinstance(last_created, (int, float)):
            break
        before_cursor = int(last_created) - 1
        time.sleep(sleep_s)

if __name__ == "__main__":
    after_epoch = to_epoch(AFTER_DATE)
    fetch_all_after(SUBREDDIT, after_epoch)
