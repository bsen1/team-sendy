#!/usr/bin/env python3
"""
PullPush (Pushshift mirror) → text chunks AFTER a chosen date
+ automatic “Latest Campus Digest” tuned for r/UCSantaBarbara.

Relevancy = tanh(log1p(score)) * (exp(-ln 2 * age_days / HALF_LIFE_DAYS))**FRESHNESS_ALPHA
- Bounded to [0, 1]
- Gentler recency via FRESHNESS_ALPHA (default 0.45)

TSV Output (printed first):
  relevancy_score    text

Each post becomes:
  - Title → one chunk
  - Body  → split into paragraph chunks (blank-line delimiter)
  - All chunks from the same post share the same relevancy score

After the TSV, the script prints a “Latest Campus Digest”
generated via embeddings + clustering (UMAP + HDBSCAN).
"""

import math
import re
import time
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any

# ===================== STAGE 1: FETCH / CLEAN / RELEVANCY (UCSB-tuned) =====================

# ---- CONFIG (edit these only if needed) ----
SUBREDDIT        = "UCSantaBarbara"
AFTER_DATE       = "2025-05-01"     # "YYYY-MM-DD" or epoch seconds string
HALF_LIFE_DAYS   = 10.0             # emphasize ~last 1–2 weeks
FRESHNESS_ALPHA  = 0.45             # softens recency dominance
MIN_BODY_CHARS   = 8                # drop body chunks shorter than this (titles are kept)
# -------------------------------------------------------------------------------------------

# Fixed 'today' (UTC) for age calculation (matches your crawl cutoff date)
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
        # ensure ints
        if "after" in params and params["after"] is not None:
            params["after"] = int(params["after"])
        if "before" in params and params["before"] is not None:
            params["before"] = int(params["before"])
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 6.0)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 6.0)
    return []

def sanitize_for_tsv(s: str) -> str:
    """Strip and collapse newlines/tabs to spaces for clean TSV output."""
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_paragraphs(markdown_text: str) -> List[str]:
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
    freshness  = freshness ** float(FRESHNESS_ALPHA)  # soften recency dominance

    r = engagement * freshness
    # ensure strict bounds
    if r < 0.0: r = 0.0
    if r > 1.0: r = 1.0
    return r

# ===================== STAGE 2: FROM CHUNKS → "LATEST CAMPUS DIGEST" (UCSB-tuned) =====================

# ---- CONFIG (no CLI) ----
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_LEN       = 360          # UCSB posts are concise; trims rambling rants
TOP_K_CLUSTERS     = 6            # dining, housing, safety, classes/advising, sports, events
TOP_SNIPPETS_PER   = 4
KEYWORDS_PER_CLUST = 5
DROP_LINK_ONLY     = True
DROP_SHORT_BODY    = True
SEED               = 42

# UMAP/HDBSCAN knobs tuned for mid-size, slang-y college text
UMAP_N_NEIGHBORS   = 12
UMAP_MIN_DIST      = 0.0
UMAP_N_COMPONENTS  = 10

def _pick_min_cluster_size(n_rows: int) -> int:
    """
    UCSB adaptive clustering:
    - small day (<800 chunks): allow tighter topics
    - medium (<=1600): moderate
    - heavy: larger to avoid fragments
    """
    if n_rows < 800:    return 6
    if n_rows < 1600:   return 8
    return 12

MIN_SAMPLES = None  # let HDBSCAN pick based on density

# ---- Lightweight guards: import optional deps only when digest runs ----
def _lazy_imports():
    from sentence_transformers import SentenceTransformer
    import numpy as np, pandas as pd
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return SentenceTransformer, np, pd, umap, hdbscan, TfidfVectorizer, ENGLISH_STOP_WORDS

# --- Helpers for Stage 2 ---
_URL_ONLY_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
def _strip_urls(s: str) -> str:
    return re.sub(r"(https?://\S+|www\.\S+)", "", s or "").strip()

def is_link_only(s: str) -> bool:
    s2 = (s or "").strip()
    if not s2:
        return False
    return bool(_URL_ONLY_RE.match(s2)) or (len(_strip_urls(s2)) == 0)

def make_keywords(texts: List[str], top_n: int = 6) -> List[str]:
    SentenceTransformer, np, pd, _, _, TfidfVectorizer, ENGLISH_STOP_WORDS = _lazy_imports()
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=ENGLISH_STOP_WORDS, max_features=4096)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    # keep non-trivial terms
    return [t for t in terms[order][:top_n] if len(t) > 2]

def collect_rows(subreddit: str, after_epoch: int) -> List[Dict[str, Any]]:
    """
    Fetch posts and:
      - print TSV exactly as requested (relevancy_score \t text)
      - collect (rel, text, ts) rows for clustering digest
    """
    session = requests.Session(); session.headers.update(HEADERS)
    params = {
        "subreddit": subreddit,
        "size": 250,
        "sort": "desc",
        "sort_type": "created_utc",
        "after": int(after_epoch),
    }
    before_cursor = None
    rows: List[Dict[str, Any]] = []

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

            # Title chunk — always kept unless link-only
            title_raw = (post.get("title") or "").strip()
            if title_raw:
                title_chunk = sanitize_for_tsv(title_raw)
                if title_chunk and (not DROP_LINK_ONLY or not is_link_only(title_chunk)):
                    print(f"{rscore:.6f}\t{title_chunk}")
                    rows.append({"rel": rscore, "text": title_chunk[:MAX_TEXT_LEN], "ts": int(created_ts)})

            # Body → split into paragraph chunks, filter out super-short and link-only
            body_raw = post.get("selftext") or ""
            if body_raw in ("[deleted]", "[removed]"):
                body_raw = ""
            paragraphs = split_paragraphs(body_raw)

            for para in paragraphs:
                chunk = sanitize_for_tsv(para)
                if DROP_SHORT_BODY and len(chunk) < MIN_BODY_CHARS:
                    continue
                if DROP_LINK_ONLY and is_link_only(chunk):
                    continue
                print(f"{rscore:.6f}\t{chunk}")
                rows.append({"rel": rscore, "text": chunk[:MAX_TEXT_LEN], "ts": int(created_ts)})

        last_created = data[-1].get("created_utc")
        if not isinstance(last_created, (int, float)):
            break
        before_cursor = int(last_created) - 1
        time.sleep(0.2)

    return rows

def cluster_and_digest(rows: List[Dict[str, Any]]):
    """Embed → reduce → cluster → rank clusters → print a compact digest."""
    SentenceTransformer, np, pd, umap_mod, hdbscan_mod, _, _ = _lazy_imports()

    if not rows:
        print("\n# Latest Campus Digest\n(no posts after the chosen date)")
        return

    df = pd.DataFrame(rows)

    # Recentness separate from relevancy (which already includes freshness)
    age_days = (FIXED_TODAY_TS - df["ts"].values) / 86400.0
    recent_w = np.exp(-np.log(2) * age_days / HALF_LIFE_DAYS) ** FRESHNESS_ALPHA

    # Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)

    # UMAP (slightly fewer neighbors for crisper UCSB topics)
    um = umap_mod.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric="cosine",
        random_state=SEED
    )
    emb_um = um.fit_transform(emb)

    # HDBSCAN with adaptive min_cluster_size
    min_cluster_size = _pick_min_cluster_size(len(df))
    clusterer = hdbscan_mod.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(emb_um)
    df["cluster"] = labels

    # Score clusters by: size, mean relevance, and “right-now” weight
    def cluster_score(g):
        size = len(g)
        mean_rel = g["rel"].mean()
        mean_recent = recent_w[g.index].mean()
        # UCSB weighting: relevance a bit more important than raw size
        return (size ** 0.30) * (mean_rel ** 0.50) * (mean_recent ** 0.35)

    valid = df[df["cluster"] != -1].copy()
    if valid.empty:
        # Fallback: treat everything as one “misc” item
        valid = df.copy()
        valid["cluster"] = 0

    ranking = (
        valid.groupby("cluster")
             .apply(cluster_score)
             .sort_values(ascending=False)
             .head(TOP_K_CLUSTERS)
    )

    # Print digest
    print("\n# Latest Campus Digest\n")
    for cid in ranking.index.tolist():
        sub = valid[valid["cluster"] == cid].copy().sort_values("rel", ascending=False).head(TOP_SNIPPETS_PER)

        # Headline keywords
        kws = make_keywords(sub["text"].tolist(), top_n=KEYWORDS_PER_CLUST)
        headline = (", ".join(kws[:2]) if kws else sub.iloc[0]["text"])[:88]

        # When (median UTC time from cluster)
        import numpy as _np  # local to avoid name shadow
        med_ts = int(_np.median(sub["ts"].values))
        when = datetime.utcfromtimestamp(med_ts).strftime("%b %d, %H:%M UTC")

        # Representative bullets
        bullets = []
        for t in sub["text"].head(4):
            s = t.strip()
            if len(s) > 160:
                s = s[:157] + "…"
            bullets.append(f"- {s}")

        print(f"## {headline}\n_{when}_")
        for b in bullets:
            print(b)
        print("")  # blank line between items

# ---- MAIN ----
if __name__ == "__main__":
    after_epoch = to_epoch(AFTER_DATE)
    rows = collect_rows(SUBREDDIT, after_epoch)  # prints TSV first
    cluster_and_digest(rows)                     # then prints digest
