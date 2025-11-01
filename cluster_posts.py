#!/usr/bin/env python3
"""
Post Clustering Script using HuggingFace Sentence Transformers
- Loads posts from TSV
- Generates embeddings with sentence-transformers
- Clusters and analyzes topics
- Uses GPT (unwrap_openai) to create headlines (multiple per cluster allowed)
- Selects a diverse set of 3–5 headlines (by list index, not cluster id)
- Saves outputs to files (no cluster labels in the output file)
- Safely writes selected headlines to Supabase:
    * Insert new headline_info rows
    * Insert subreddit mappings
    * Delete stale rows not in the new set

Env:
  SUPABASE_ANON_KEY=...
  SUBREDDIT=UCSD                    # optional; defaults to 'UCSD'
"""

import os
import re
import warnings
import asyncio
import time
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from supabase import create_client

from unwrap_openai import create_openai_completion, GPT5Deployment, ReasoningEffort

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ----- Config -----
SUBREDDIT = os.getenv("SUBREDDIT")

# Supabase configuration
SUPABASE_URL = "https://yhcpqdbudjchqkcacsnl.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # put this in your .env

def get_supabase_client():
    """Initialize and return Supabase client."""
    if not SUPABASE_KEY:
        raise ValueError("SUPABASE_ANON_KEY not found in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------
# Minimal LLM HTTP error handling
# --------------------------
def _retryable_exc_str(e: Exception) -> bool:
    """
    Heuristic: retry on typical transient transport/status errors.
    We keep this intentionally simple and string-based to avoid importing httpx types.
    """
    s = str(e).lower()
    retry_markers = [
        "timeout", "timed out", "temporarily unavailable",
        "connection error", "connecterror", "readerror",
        "server error", "502", "503", "504", "bad gateway",
        "service unavailable", "gateway timeout", "rate limit", "429",
    ]
    return any(m in s for m in retry_markers)

async def _call_llm_with_retry(*, messages, model, reasoning_effort, max_completion_tokens,
                               retries: int = 1, backoff: float = 1.0):
    """
    Minimal wrapper: try once + 1 retry on transient HTTP/transport errors.
    Returns the response object or None.
    """
    for attempt in range(retries + 1):
        try:
            return await create_openai_completion(
                messages=messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens
            )
        except Exception as e:
            if attempt < retries and _retryable_exc_str(e):
                time.sleep(backoff)
                continue
            return None

# --------------------------
# Data loading & preprocessing
# --------------------------
def load_posts(file_path: str):
    """Load posts from TSV with columns at least: title, body, score."""

    df = pd.read_csv(file_path, sep='\t')



    required = {'title', 'body', 'score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(list(missing))}")

    # Combine title and body for embedding
    df['combined_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

    # Clean text: remove URLs, punctuation→space, collapse whitespace
    df['combined_text'] = df['combined_text'].astype(str)
    df['combined_text'] = df['combined_text'].apply(
        lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE)
    )
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['combined_text'] = df['combined_text'].apply(lambda x: ' '.join(x.split()))

    # Remove boilerplate phrases (optional)
    question_phrases = [
        'does anyone know', 'does anyone', 'anyone know', 'anyone have', 'anyone got',
        'anyone else', 'anyone here', 'anyone can', 'anyone want', 'anyone need',
        'does anyone have', 'does anyone else', 'does anyone want', 'does anyone need',
        'can anyone', 'will anyone', 'has anyone', 'is anyone', 'are there any',
        'looking for', 'need help', 'please help', 'any advice', 'any suggestions',
        'any recommendations', 'any tips', 'any ideas', 'any thoughts'
    ]
    def clean_common_phrases(text: str):
        t = text.lower()
        for phrase in question_phrases:
            t = t.replace(phrase, '')
        return ' '.join(t.split())

    df['combined_text'] = df['combined_text'].apply(clean_common_phrases)

    # Remove empty/very short posts
    df = df[df['combined_text'].str.len() > 10].copy()

    return df

# --------------------------
# Embeddings
# --------------------------
def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate sentence embeddings using HuggingFace sentence-transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), show_progress_bar=True)
    return embeddings

# --------------------------
# Clustering selection
# --------------------------
def test_clustering_algorithms(embeddings, optimal_k: int):
    """Test a few algorithms and return labels from the best by silhouette."""
    algorithms = {
        'KMeans': KMeans(n_clusters=optimal_k, random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
        'GaussianMixture': GaussianMixture(n_components=optimal_k, random_state=42),
        'Spectral': SpectralClustering(n_clusters=optimal_k, random_state=42)
    }

    best_algorithm = None
    best_score = -1
    results = {}

    for name, algo in algorithms.items():
        try:
            labels = algo.fit_predict(embeddings)
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                results[name] = {'labels': labels, 'score': score}
                if score > best_score:
                    best_score = score
                    best_algorithm = name
            else:
                results[name] = {'labels': labels, 'score': -1}
        except Exception:
            results[name] = {'labels': None, 'score': -1}

    if best_algorithm is None:
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(embeddings)
        return km.labels_, 'KMeans(fallback)'

    return results[best_algorithm]['labels'], best_algorithm

def find_optimal_clusters(embeddings, max_k=None):
    """Find optimal number of clusters."""
    n_samples = len(embeddings)

    if n_samples < 8:
        return 2, {'k': [2], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [], 'inertia': []}

    if max_k is None:
        min_k = 5
        max_k = max(6, min(25, n_samples // 30))
        if max_k <= min_k:
            min_k, max_k = 2, min(10, n_samples // 2)
    else:
        min_k = 5

    ks = list(range(min_k, max_k + 1))
    results = {'k': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [], 'inertia': []}

    for k in ks:
        if k < 2 or k >= n_samples:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(embeddings, labels)
        ch = calinski_harabasz_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)
        inertia = kmeans.inertia_
        results['k'].append(k)
        results['silhouette'].append(sil)
        results['calinski_harabasz'].append(ch)
        results['davies_bouldin'].append(db)
        results['inertia'].append(inertia)

    if not results['k']:
        return 2, results

    ks_arr = np.array(results['k'])
    sil_opt = ks_arr[np.argmax(results['silhouette'])]
    ch_opt  = ks_arr[np.argmax(results['calinski_harabasz'])]
    db_opt  = ks_arr[np.argmin(results['davies_bouldin'])]

    inertia = np.array(results['inertia'])
    if len(inertia) >= 3:
        diff1 = np.diff(inertia); diff2 = np.diff(diff1)
        elbow_idx = int(np.argmax(diff2)) + 2
        elbow_opt = ks_arr[min(elbow_idx, len(ks_arr) - 1)]
    else:
        elbow_opt = sil_opt

    optimal_ks = [sil_opt, ch_opt, db_opt, elbow_opt]
    k_counts = Counter(optimal_ks)
    consensus_k = k_counts.most_common(1)[0][0]
    if consensus_k < 2: consensus_k = 2

    return consensus_k, results

def cluster_posts(embeddings, n_clusters):
    """Cluster posts using best algorithm."""
    labels, best_algo = test_clustering_algorithms(embeddings, n_clusters)
    return labels, best_algo

# --------------------------
# Analysis (representativeness)
# --------------------------
def analyze_clusters(df: pd.DataFrame, cluster_labels, embeddings):
    """Analyze clusters to pick representative posts and compute topic signals."""
    df = df.copy()
    df['cluster'] = cluster_labels

    cluster_analysis = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_posts = df[df['cluster'] == cluster_id]
        all_text = ' '.join(cluster_posts['combined_text'].tolist()).lower()
        words = re.findall(r'\b\w{4,}\b', all_text)
        word_freq = Counter(words)
        top_words = [w for w, _ in word_freq.most_common(10)]

        # map DataFrame index -> embedding position
        df_index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
        positions = [df_index_to_pos[idx] for idx in cluster_posts.index]
        cluster_emb = embeddings[positions]

        # Weighted centroid (by normalized score)
        scores = cluster_posts['score'].astype(float).values
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        weights = 0.3 + 0.7 * norm
        centroid = np.average(cluster_emb, axis=0, weights=weights)

        # Distances & combined representativeness
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        semantic = 1.0 / (1.0 + distances)
        upvote = norm
        combined = 0.7 * semantic + 0.3 * upvote

        # Top N representatives
        top_idx = np.argsort(combined)[::-1][:10]
        sample_posts = cluster_posts.iloc[top_idx][['title', 'body', 'score']]

        cluster_analysis[cluster_id] = {
            'size': len(cluster_posts),
            'top_words': top_words,
            'sample_posts': sample_posts,
            'avg_score': float(cluster_posts['score'].mean()),
            'max_score': int(cluster_posts['score'].max()),
        }

    return cluster_analysis

# --------------------------
# Headline Generation (LLM)
# --------------------------
async def generate_cluster_headlines(cluster_analysis):
    """
    Generate 0..N headline ideas per cluster.
    Returns a flat list of dicts: {headline, description, meta}
    (no cluster labels in output/rendering).
    """
    ideas = []

    for cluster_id, analysis in cluster_analysis.items():
        # Build small sample (already representative)
        cluster_posts = []
        for _, post in analysis['sample_posts'].iterrows():
            cluster_posts.append({
                'title': str(post['title']),
                'body': str(post['body']),
                'score': int(post['score']),
            })

        # Truncate to keep context manageable
        def trunc(s, limit):
            s = s or ""
            return s if len(s) <= limit else s[:limit] + " …"

        posts_text = ""
        for i, p in enumerate(cluster_posts[:10], 1):
            posts_text += (
                f"Post {i}:\n"
                f"Title: {trunc(p['title'], 140)}\n"
                f"Content: {trunc(p['body'], 600)}\n"
                f"Upvotes: {p['score']}\n\n"
            )

        # (Do not change prompts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sharp, concise headline writer analyzing clusters of Reddit posts. "
                    "Each cluster may contain one or several distinct discussion topics. Generate AT LEAST ONE, PREFERABLY 2 DISTINCT headlines and description from this cluster."
                    "If multiple distinct, unrelated topics appear, generate MULTIPLE headlines — "
                    "each with its own short description.\n\n"
                    "Prioritize headlines for posts about serious campus news, controversial discussions, quirky/funny, events or announcements, student life moments, and unusual topics."
                    "Guidelines for headlines:\n"
                    "- Each headline must be UNDER 12 WORDS.\n"
                    "- DO NOT include the subreddit name or school name.\n"
                    "- Headlines should clearly describe the situation, issue, or event.\n"
                    "- Write them in a funny, casual style as if a college student wrote them.\n\n"
                    "- When writing the headlines and descriptions, feel free to use slang like lowkey, highkey, mid, hits different, vibe, valid, chill, mood, etc.\n\n"
                    "Format exactly:\n"
                    "HEADLINE 1: <headline>\n"
                    "DESCRIPTION 1: <one-paragraph description>\n\n"
                    "HEADLINE 2: <headline>\n"
                    "DESCRIPTION 2: <one-paragraph description>\n\n"
                    "(Add more if distinct topics are found.)"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Cluster context:\n{posts_text}\n\n"
                    "Generate headlines about these posts."
                ),
            },
        ]

        # Minimal HTTP-safe call
        response = await _call_llm_with_retry(
            messages=messages,
            model=GPT5Deployment.GPT_5_NANO,
            reasoning_effort=ReasoningEffort.MINIMAL,
            max_completion_tokens=300,
            retries=1,
            backoff=1.0
        )
        if not response or not getattr(response, "choices", None):
            continue

        response_text = (getattr(response.choices[0].message, "content", "") or "").strip()
        if not response_text:
            continue

        # Parse multiple numbered pairs
        pairs = []
        pattern = r"HEADLINE\s*\d*\s*:\s*(.+?)\s*(?:\r?\n)+DESCRIPTION\s*\d*\s*:\s*(.+?)(?=(?:\r?\n\s*HEADLINE|\Z))"
        for m in re.finditer(pattern, response_text, flags=re.IGNORECASE | re.DOTALL):
            h = m.group(1).strip()
            d = m.group(2).strip()
            if len(h) >= 3 and len(d) >= 10:
                pairs.append((h, d))

        # fallback single-line parse
        if not pairs and response_text:
            lines = [l.strip() for l in response_text.splitlines() if l.strip()]
            if lines:
                h = lines[0]
                d = ' '.join(lines[1:]) if len(lines) > 1 else ""
                if len(h) >= 3 and len(d) >= 10:
                    pairs.append((h, d))

        if not pairs:
            continue

        for (h, d) in pairs:
            ideas.append({
                "headline": h,
                "description": d,
                "meta": {
                    "cluster_id": cluster_id,
                    "size": analysis['size'],
                    "avg_score": analysis['avg_score'],
                    "max_score": analysis['max_score'],
                    "top_words": analysis['top_words'][:5],
                }
            })

    return ideas

# --------------------------
# Headline Selection (LLM) – by list index
# --------------------------
async def select_most_interesting_headlines(all_ideas):
    """
    Select a diverse set of 3–5 headlines from a flat list of ideas (no cluster labels).
    Returns a list of {headline, description, meta}.
    """
    # Deduplicate by headline text
    seen = set()
    candidates = []
    for idea in all_ideas:
        h = idea["headline"].strip()
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(idea)

    # --- Added: print candidate headlines ---
    if candidates:
        print("\n📝 Candidate headlines:")
        for i, cand in enumerate(candidates, 1):
            print(f"{i}. {cand['headline']}")

    if len(candidates) < 3:
        return []

    # Build numbered list for LLM (indices only)
    lines = []
    for idx, idea in enumerate(candidates, 1):
        tw = ', '.join(idea.get("meta", {}).get("top_words", []))
        size = idea.get("meta", {}).get("size")
        avg = idea.get("meta", {}).get("avg_score")
        mx = idea.get("meta", {}).get("max_score")
        lines.append(
            f"{idx}. {idea['headline']}\n"
            f"   - {idea['description']}\n"
            f"   - signals: size={size}, avg={avg:.1f} if avg is not None else 'n/a', max={mx}, keywords=[{tw}]"
        )
    headlines_text = "\n\n".join(lines)

    # (Do not change prompts)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert content curator selecting headlines from a list of options."
                "Your goal is to capture a DIVERSE and ENGAGING mix of content — not just the most important items. "
                "Choose 5 to 7 eye-catching headlines that represent some of the following topics: "
                "serious news, controversial discussions, outlandish posts, campus events, quirky stuff, memes.\n\n"
                "Avoid redundancy. Respond ONLY with the chosen NUMBERS (indices), separated by commas, "
                "in order of most to least interesting (e.g., '4, 1, 9, 2')."
            )
        },
        {
            "role": "user",
            "content": f"""Here are candidate headlines (numbered):

{headlines_text}

Select a diverse, interesting mix of headlines. 
Respond ONLY with the numbers (e.g., '4, 1, 9, 2'):"""
        }
    ]

    response = await _call_llm_with_retry(
        messages=messages,
        model=GPT5Deployment.GPT_5_NANO,
        reasoning_effort=ReasoningEffort.MINIMAL,
        max_completion_tokens=50,
        retries=1,
        backoff=1.0
    )
    if not response or not getattr(response, "choices", None):
        k = min(5, max(3, len(candidates)))
        chosen = candidates[:k]
        # --- Added: print selected headlines ---
        print("\n🎯 Selected headlines:")
        for i, h in enumerate(chosen, 1):
            print(f"{i}. {h['headline']}")
        return chosen

    selected_text = (getattr(response.choices[0].message, "content", "") or "").strip()
    if not selected_text:
        k = min(5, max(3, len(candidates)))
        chosen = candidates[:k]
        print("\n🎯 Selected headlines:")
        for i, h in enumerate(chosen, 1):
            print(f"{i}. {h['headline']}")
        return chosen

    try:
        idxs = [int(x.strip()) for x in selected_text.split(",") if x.strip().isdigit()]
        idxs = [i for i in idxs if 1 <= i <= len(candidates)]
        chosen = [candidates[i - 1] for i in idxs]
        if not chosen:
            k = min(5, max(3, len(candidates)))
            chosen = candidates[:k]
        # --- Added: print selected headlines ---
        print("\n🎯 Selected headlines:")
        for i, h in enumerate(chosen, 1):
            print(f"{i}. {h['headline']}")
        return chosen
    except Exception:
        k = min(5, max(3, len(candidates)))
        chosen = candidates[:k]
        print("\n🎯 Selected headlines:")
        for i, h in enumerate(chosen, 1):
            print(f"{i}. {h['headline']}")
        return chosen

# --------------------------
# Output
# --------------------------
def save_headlines_to_file(selected_ideas, filename="cluster_headlines.txt"):
    """Save chosen headlines to file; no cluster labels in output."""
    with open(filename, 'w') as f:
        f.write(f"{SUBREDDIT.upper()} REDDIT — SELECTED HEADLINES\n")
        f.write("="*60 + "\n\n")
        for i, idea in enumerate(selected_ideas, 1):
            f.write(f"{i}. {idea['headline']}\n")
            f.write(f"   {idea['description']}\n\n")

# --------------------------
# Supabase (safe write)
# --------------------------
async def store_headlines_in_supabase(selected_ideas):
    """
    Store the chosen headlines in Supabase.
    SAFETY: Do not delete existing rows unless we have new inserts ready.
    """
    try:
        if not selected_ideas:
            return

        supabase = get_supabase_client()

        # Prepare new rows
        new_rows = [{"headline": i["headline"], "description": i["description"]} for i in selected_ideas]
        inserted = supabase.table('headline_info').insert(new_rows).execute()
        if not inserted.data:
            return
        new_ids = [row['headline_id'] for row in inserted.data]

        map_rows = [{'subreddit': SUBREDDIT, 'headline_id': hid} for hid in new_ids]
        mapped = supabase.table('subreddit_to_headline_id').insert(map_rows).execute()
        if not mapped.data:
            supabase.table('headline_info').delete().in_('headline_id', new_ids).execute()
            return

        # Now delete any older mappings/headlines for this subreddit not in new_ids
        existing = supabase.table('subreddit_to_headline_id') \
                           .select('headline_id') \
                           .eq('subreddit', SUBREDDIT) \
                           .execute()

        keep = set(new_ids)
        old_ids = [r['headline_id'] for r in (existing.data or []) if r['headline_id'] not in keep]

        if old_ids:
            supabase.table('subreddit_to_headline_id') \
                .delete() \
                .eq('subreddit', SUBREDDIT) \
                .in_('headline_id', old_ids) \
                .execute()
            supabase.table('headline_info').delete().in_('headline_id', old_ids).execute()

    except Exception:
        return

# --------------------------
# Orchestration
# --------------------------
async def main():
    df = load_posts(f"{SUBREDDIT.lower()}.tsv")
    embeddings = generate_embeddings(df['combined_text'].tolist())

    optimal_k, _ = find_optimal_clusters(embeddings)
    optimal_k = max(optimal_k, 7)
    # --- Added: print the final k chosen ---
    print(f"\n🔢 optimal_k chosen: {optimal_k}")

    cluster_labels, _ = cluster_posts(embeddings, optimal_k)

    cluster_analysis = analyze_clusters(df, cluster_labels, embeddings)

    all_ideas = await generate_cluster_headlines(cluster_analysis)
    selected_ideas = await select_most_interesting_headlines(all_ideas)

    save_headlines_to_file(selected_ideas, filename="cluster_headlines.txt")
    await store_headlines_in_supabase(selected_ideas)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    df_with_clusters.to_csv('posts_with_clusters.tsv', sep='\t', index=False)

if __name__ == "__main__":
    asyncio.run(main())
