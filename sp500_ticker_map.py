# sp500_ticker_map.py
# ------------------------------------------------------------
# Build a dictionary mapping S&P 500 tickers to:
#   { TICKER: (Parsed Company Name, [Parsed, Name, Split]) }
#
# Cleans:
# - Share-class noise: "Class A/B/C…" (parenthetical or standalone)
# - Trailing/stacked legal suffixes: Inc., Corp, Co., LLC, Ltd, PLC, etc.
# - Generic business nouns: Holdings, Group, Services, Financial, etc.
# - Definite-article variants: leading "The ", "(The)" at end, ", The" at end
#
# Includes 403-friendly headers, retry logic, and pandas parsing.
# ------------------------------------------------------------

from __future__ import annotations
import re
import time
from typing import Dict, List, Tuple
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, UTC

# Wikipedia source
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/List_of_S%26P_500_companies"

# Descriptive User-Agent to avoid 403 (replace contact with your email if you like)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36 (contact: you@example.com)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# --- Name cleaning helpers ----------------------------------------------------

# Remove any parenthetical that contains the word "class"
#   e.g., "Alphabet Inc. (Class C)" -> "Alphabet Inc."
_RE_PAREN_CLASS = re.compile(r"\s*\((?=[^)]*class\b)[^)]*\)", flags=re.IGNORECASE)

# Remove standalone occurrences like "Class A", "Class B voting", "Class C Common Stock"
_RE_CLASS_WORD = re.compile(r"\bclass\s+[A-Za-z0-9\-]+\b", flags=re.IGNORECASE)

# Leading article (common cleanup)
_RE_LEADING_THE = re.compile(r"^\s*the\s+", flags=re.IGNORECASE)

# Wikipedia sometimes encodes leading "The" as a trailing token:
#   "Cooper Companies (The)" or "Cooper Companies, The"
_RE_PAREN_THE_ANYWHERE = re.compile(r"\(\s*the\s*\)", flags=re.IGNORECASE)
_RE_TRAIL_COMMA_THE = re.compile(r",\s*the\s*$", flags=re.IGNORECASE)

# Common corporate/legal entity suffixes to trim from the END of the name.
# Handles commas, periods, and stacked forms like "Company, Inc." or "Co., Ltd."
_LEGAL_SUFFIXES = (
    r"incorporated|inc|corp|corporation|co|company|"
    r"ltd|limited|plc|llc|llp|lp|l\.p\.|"
    r"s\.a\.|s\.a\.b\.?(?:\s+de\s+c\.v\.)?|s\.p\.a\.|"
    r"n\.v\.|ag|se|oyj|aps|a\/s|k\.k\.|kabushiki\s+kaisha"
)
_RE_TRAILING_LEGAL = re.compile(
    rf"(?:\s*,?\s+(?:{_LEGAL_SUFFIXES})(?:\.|\b))+\s*$",
    flags=re.IGNORECASE,
)

# Generic “business nouns” to de-noise (anywhere in the string).
# Keep the list conservative to avoid destroying meaningful names.
_GENERIC_BUSINESS_TERMS = (
    r"holdings?|group(?:s)?|services?|solutions?|systems?|industries?|"
    r"resources?|partners?|networks?|brands?|communications?|properties?|"
    r"financial|finance|technologies|technology|enterprises?|ventures?"
)
_RE_GENERIC_WORDS = re.compile(
    rf"\b(?:{_GENERIC_BUSINESS_TERMS})\b", flags=re.IGNORECASE
)

# Extra whitespace / punctuation cleanup
_RE_SPACES = re.compile(r"\s{2,}")

def _strip_trailing_legal(name: str) -> str:
    """Remove trailing legal suffixes iteratively (e.g., ', Inc.', ', Co., Ltd.')."""
    s = name
    while True:
        new = _RE_TRAILING_LEGAL.sub("", s)
        if new == s:
            return s.strip()
        s = new.strip()

def _remove_generic_business_words(name: str) -> str:
    """
    Remove common generic business nouns anywhere in the string.
    We remove whole words (case-insensitive). If removal would empty
    the name, we leave the original token to avoid blank results.
    """
    tokens = re.split(r"(\s+|&|/)", name)  # keep separators like spaces/&/ for readability
    cleaned = []
    for tok in tokens:
        if tok and not re.fullmatch(r"(\s+|&|/)", tok):
            if _RE_GENERIC_WORDS.fullmatch(tok):
                continue  # drop generic word
        cleaned.append(tok)

    candidate = "".join(cleaned).strip()
    if not candidate:
        return name.strip()
    return candidate

def _final_punct_whitespace_trim(s: str) -> str:
    s = _RE_SPACES.sub(" ", s).strip()
    # Trim dangling punctuation/commas/hyphens/periods
    s = s.strip(" ,;:-.")
    return s

def _clean_company_name(name: str) -> str:
    """Return a concise, brand-identifiable company name."""
    if not name:
        return name
    s = name

    # 0) normalize Wikipedia-style trailing 'The'
    #    e.g., "Cooper Companies (The)" -> "Cooper Companies"
    #          "Cooper Companies, The"  -> "Cooper Companies"
    s = _RE_PAREN_THE_ANYWHERE.sub("", s)
    s = _RE_TRAIL_COMMA_THE.sub("", s)

    # 1) drop any (...) that mention class
    s = _RE_PAREN_CLASS.sub("", s)

    # 2) drop standalone "Class <X>" phrases
    s = _RE_CLASS_WORD.sub("", s)

    # 3) trim trailing legal entity designators
    s = _strip_trailing_legal(s)

    # 4) drop leading "The" (if any remains)
    s = _RE_LEADING_THE.sub("", s)

    # 5) remove generic business nouns (holdings, group, services, financial, etc.)
    s = _remove_generic_business_words(s)

    # 6) collapse leftovers and trim punctuation
    s = _final_punct_whitespace_trim(s)
    return s

def _name_to_tokens(cleaned: str) -> List[str]:
    """
    Split cleaned name into tokens like ["Chipotle", "Mexican", "Grill"].
    Keeps alphanumerics and simple word parts; drops stray punctuation and symbols.
    """
    # Split on whitespace, then strip punctuation from each token; keep alphanumerics, + . ' -
    raw = cleaned.split()
    tokens: List[str] = []
    for t in raw:
        # Remove surrounding punctuation
        core = re.sub(r"^[^\w'+.-]+|[^\w'+.-]+$", "", t)
        if core:
            tokens.append(core)
    return tokens

# --- Fetch / parse Wikipedia --------------------------------------------------

def _fetch_html_with_retry(url: str, max_tries: int = 3, backoff: float = 0.8) -> str:
    """Fetch HTML with retries to handle occasional 403s or timeouts."""
    last_err = None
    for i in range(max_tries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed to fetch HTML from {url}: {last_err}")

def _load_sp500_mapping() -> Dict[str, Tuple[str, List[str]]]:
    """
    Scrape the S&P 500 constituents table and return:
        { ticker: (cleaned_company_name, [tokens]) }
    """
    try:
        html = _fetch_html_with_retry(_WIKI_URL)
    except Exception:
        html = _fetch_html_with_retry(_WIKI_REST_HTML)

    # Wrap literal HTML in StringIO to avoid FutureWarning in pandas
    tables = pd.read_html(StringIO(html))

    df = None
    for tbl in tables:
        cols = {str(c).strip().lower() for c in tbl.columns}
        if {"symbol", "security"} <= cols:
            df = tbl
            break

    if df is None:
        # Fallback: pick the largest table if the expected columns weren't found
        df = max(tables, key=lambda t: t.shape[0]) if tables else None
    if df is None:
        raise RuntimeError("Could not locate S&P 500 table in Wikipedia HTML.")

    # Normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]
    sym_col = "symbol" if "symbol" in df.columns else df.columns[0]
    name_col = "security" if "security" in df.columns else df.columns[1]

    mapping: Dict[str, Tuple[str, List[str]]] = {}
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip().upper()
        raw = str(row[name_col]).strip()
        cleaned = _clean_company_name(raw)
        if sym and cleaned and sym != "NAN":
            tokens = _name_to_tokens(cleaned)
            # Ensure we don’t end up with empty token lists; if so, fall back to a simple split on spaces
            if not tokens:
                tokens = [t for t in cleaned.split() if t]
            mapping[sym] = (cleaned, tokens)
    return mapping

# --- Build mapping on import --------------------------------------------------

TICKER_TO_NAME: Dict[str, Tuple[str, List[str]]] = _load_sp500_mapping()
BUILT_AT_UTC: str = datetime.now(UTC).isoformat(timespec="seconds")

__all__ = ["TICKER_TO_NAME", "BUILT_AT_UTC"]

if __name__ == "__main__":
    print(f"Built at {BUILT_AT_UTC}")
    print(f"Total tickers: {len(TICKER_TO_NAME)}")

    # Quick demo examples if present:
    for t in ["COO", "CME", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "BRK.B"]:
        if t in TICKER_TO_NAME:
            print(f"{t}: {TICKER_TO_NAME[t]}")

    # Print full mapping
    print("\nFull mapping:")
    for ticker in sorted(TICKER_TO_NAME):
        name, parts = TICKER_TO_NAME[ticker]
        print(f"{ticker}: ({name}, {parts})")
