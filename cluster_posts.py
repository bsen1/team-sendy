#!/usr/bin/env python3
"""
Post Clustering Script using HuggingFace Sentence Transformers
This script loads posts from posts.tsv, generates embeddings, and clusters them to identify topics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import re
import warnings
from collections import Counter
import os
from dotenv import load_dotenv
from unwrap_openai import create_openai_completion, GPT5Deployment, ReasoningEffort, generate_embeddings_batch, EmbeddingModel
import asyncio
from supabase import create_client, Client

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = "https://yhcpqdbudjchqkcacsnl.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # You'll need to add this to your .env file

def get_supabase_client():
    """Initialize and return Supabase client."""
    if not SUPABASE_KEY:
        raise ValueError("SUPABASE_ANON_KEY not found in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def load_posts(file_path):
    """Load posts from TSV file and clean the data."""
    print("Loading posts from TSV file...")
    df = pd.read_csv(file_path, sep='\t')
    
    # Combine title and body for embedding
    df['combined_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
    
    # Clean text: remove URLs, special characters, and extra whitespace
    df['combined_text'] = df['combined_text'].apply(lambda x: 
        re.sub(r'http\S+|www\S+|https\S+', '', str(x), flags=re.MULTILINE) if pd.notna(x) else '')
    df['combined_text'] = df['combined_text'].apply(lambda x: 
        re.sub(r'[^\w\s]', ' ', str(x)) if pd.notna(x) else '')
    df['combined_text'] = df['combined_text'].apply(lambda x: 
        ' '.join(str(x).split()) if pd.notna(x) else '')
    
    # Remove common question phrases that don't add topic information
    question_phrases = [
        'does anyone know', 'does anyone', 'anyone know', 'anyone have', 'anyone got',
        'anyone else', 'anyone here', 'anyone can', 'anyone want', 'anyone need',
        'does anyone have', 'does anyone else', 'does anyone want', 'does anyone need',
        'can anyone', 'will anyone', 'has anyone', 'is anyone', 'are there any',
        'looking for', 'need help', 'please help', 'any advice', 'any suggestions',
        'any recommendations', 'any tips', 'any ideas', 'any thoughts'
    ]
    
    def clean_common_phrases(text):
        text_lower = text.lower()
        for phrase in question_phrases:
            text_lower = text_lower.replace(phrase, '')
        return text_lower
    
    df['combined_text'] = df['combined_text'].apply(clean_common_phrases)
    
    # Remove empty posts
    df = df[df['combined_text'].str.len() > 10]
    
    print(f"Loaded {len(df)} posts")
    return df

async def generate_embeddings(texts):
    """Generate embeddings for a list of texts using Azure OpenAI."""
    print(f"Generating embeddings using Azure OpenAI text-embedding-3-small...")
    
    embeddings = await generate_embeddings_batch(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def test_clustering_algorithms(embeddings, optimal_k):
    """Test different clustering algorithms and return the best one."""
    # Testing clustering algorithms silently
    
    algorithms = {
        'KMeans': KMeans(n_clusters=optimal_k, random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
        'GaussianMixture': GaussianMixture(n_components=optimal_k, random_state=42),
        'Spectral': SpectralClustering(n_clusters=optimal_k, random_state=42)
    }
    
    best_algorithm = None
    best_score = -1
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            if name == 'GaussianMixture':
                # GMM returns probabilities, need to get cluster assignments
                cluster_labels = algorithm.fit_predict(embeddings)
            else:
                cluster_labels = algorithm.fit_predict(embeddings)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(embeddings, cluster_labels)
                results[name] = {'labels': cluster_labels, 'score': score}
                # Algorithm tested silently
                
                if score > best_score:
                    best_score = score
                    best_algorithm = name
            else:
                print(f"  {name}: failed (only 1 cluster)")
                results[name] = {'labels': cluster_labels, 'score': -1}
                
        except Exception as e:
            print(f"  {name}: failed - {str(e)}")
            results[name] = {'labels': None, 'score': -1}
    
    # Best algorithm selected silently
    return results[best_algorithm]['labels'], best_algorithm

def find_optimal_clusters(embeddings, max_k=None):
    """Find optimal number of clusters using multiple methods."""
    # Finding optimal clusters silently
    
    n_samples = len(embeddings)
    
    # More fine-grained range for detailed topic clustering
    if max_k is None:
        # Focus on 5-25 clusters for fine-grained topic separation
        min_k = 5
        max_k = min(25, n_samples // 30)  # More clusters for detailed topics
    else:
        min_k = 5

    # Testing k range silently
    
    k_range = range(min_k, max_k + 1)
    
    # Store results for different methods
    results = {
        'k': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
        'inertia': []
    }
    
    # Evaluating clusters silently
    for k in k_range:
        # Testing k silently
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        # Check if we have more than 1 cluster for silhouette score
        n_clusters = len(set(cluster_labels))
        if n_clusters > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette_avg = -1  # Invalid score for single cluster
        
        calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        inertia = kmeans.inertia_
        
        results['k'].append(k)
        results['silhouette'].append(silhouette_avg)
        results['calinski_harabasz'].append(calinski_harabasz)
        results['davies_bouldin'].append(davies_bouldin)
        results['inertia'].append(inertia)
        
        # Silhouette score calculated silently
    
    # Find optimal k using different methods
    # Filter out invalid silhouette scores (-1)
    valid_silhouette_indices = [i for i, score in enumerate(results['silhouette']) if score > -1]
    if valid_silhouette_indices:
        valid_silhouette_scores = [results['silhouette'][i] for i in valid_silhouette_indices]
        best_silhouette_idx = valid_silhouette_indices[np.argmax(valid_silhouette_scores)]
        silhouette_optimal = k_range[best_silhouette_idx]
    else:
        silhouette_optimal = k_range[0]  # Fallback to first k
    
    calinski_optimal = k_range[np.argmax(results['calinski_harabasz'])]
    davies_optimal = k_range[np.argmin(results['davies_bouldin'])]  # Lower is better for Davies-Bouldin
    
    # Elbow method for inertia
    # Calculate second derivative to find elbow
    inertia_diff = np.diff(results['inertia'])
    inertia_diff2 = np.diff(inertia_diff)
    elbow_idx = np.argmax(inertia_diff2) + 2  # +2 because of double diff
    elbow_optimal = k_range[min(elbow_idx, len(k_range) - 1)]
    
    print(f"\nOptimal k by different methods:")
    print(f"  Silhouette Score: k={silhouette_optimal} (score={max(results['silhouette']):.3f})")
    print(f"  Calinski-Harabasz: k={calinski_optimal} (score={max(results['calinski_harabasz']):.3f})")
    print(f"  Davies-Bouldin: k={davies_optimal} (score={min(results['davies_bouldin']):.3f})")
    print(f"  Elbow Method: k={elbow_optimal}")
    
    # Consensus approach: take the most common optimal k, but prefer higher k for fine-grained analysis
    optimal_ks = [silhouette_optimal, calinski_optimal, davies_optimal, elbow_optimal]
    k_counts = Counter(optimal_ks)
    consensus_k = k_counts.most_common(1)[0][0]
    
    # For fine-grained analysis, ensure we have enough clusters
    if consensus_k < 8:
        consensus_k = min(8, max_k)
    
    print(f"🎯 Selected {consensus_k} clusters for analysis")
    
    return consensus_k, results

def cluster_posts(embeddings, n_clusters):
    """Cluster posts using the best algorithm."""
    print(f"Clustering posts into {n_clusters} clusters...")
    
    # Test different clustering algorithms and use the best one
    cluster_labels, best_algorithm = test_clustering_algorithms(embeddings, n_clusters)
    
    print(f"Using {best_algorithm} clustering with {n_clusters} clusters...")
    
    return cluster_labels, best_algorithm

def analyze_clusters(df, cluster_labels, embeddings):
    """Analyze clusters to identify specific topics."""
    print("Analyzing clusters...")
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Analyze each cluster
    cluster_analysis = {}
    
    for cluster_id in range(max(cluster_labels) + 1):
        cluster_posts = df[df['cluster'] == cluster_id]
        
        # Get most common words in this cluster
        all_text = ' '.join(cluster_posts['combined_text'].tolist()).lower()
        words = re.findall(r'\b\w{4,}\b', all_text)
        
        # Use all words for more nuanced topic identification
        # No stop word filtering to capture more context and subtle topics
        word_freq = Counter(words)
        top_words = [word for word, freq in word_freq.most_common(10)]
        
        # Get the most representative posts (considering both semantic similarity and upvotes)
        # Create a mapping from DataFrame index to embedding array position
        df_index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
        cluster_positions = [df_index_to_pos[idx] for idx in cluster_posts.index]
        cluster_embeddings = embeddings[cluster_positions]
        
        # Calculate weighted centroid (upvotes influence the centroid)
        scores = cluster_posts['score'].values
        # Normalize scores to 0-1 range, then add small base weight
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        weights = 0.3 + 0.7 * normalized_scores  # Base weight 0.3, up to 1.0 for high scores
        
        # Weighted centroid calculation
        weighted_centroid = np.average(cluster_embeddings, axis=0, weights=weights)
        
        # Calculate distances from each post to the weighted centroid
        distances = np.linalg.norm(cluster_embeddings - weighted_centroid, axis=1)
        
        # Combine semantic similarity (70%) with upvote score (30%)
        semantic_score = 1 / (1 + distances)  # Convert distance to similarity
        upvote_score = normalized_scores
        combined_score = 0.7 * semantic_score + 0.3 * upvote_score
        
        # Get the 10 posts with highest combined score
        best_indices = np.argsort(combined_score)[::-1][:10]
        sample_posts = cluster_posts.iloc[best_indices][['title', 'body', 'score']]
        
        # Let the clustering discover topics naturally - no hard-coded keywords
        # The topic is determined by the most common meaningful words in the cluster
        primary_topic = 'discovered_topic'
        
        # Calculate score statistics
        avg_score = cluster_posts['score'].mean()
        max_score = cluster_posts['score'].max()
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_posts),
            'top_words': top_words,
            'sample_posts': sample_posts,
            'primary_topic': primary_topic,
            'avg_score': avg_score,
            'max_score': max_score
        }
    
    return cluster_analysis

async def generate_cluster_headlines(cluster_analysis):
    """Generate interesting headlines for each cluster using OpenAI GPT."""
    print("\n🤖 Generating headlines with OpenAI GPT...")
    
    headlines = {}
    
    for cluster_id, analysis in cluster_analysis.items():
        # Prepare the cluster data for GPT
        cluster_posts = []
        for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
            cluster_posts.append({
                'title': post['title'],
                'body': post['body'],
                'score': post['score'],
                'link': f"https://reddit.com/r/UCSD"  # Generic link since we don't have specific post IDs
            })
        
        # Create prompt for GPT
        posts_text = ""
        for i, post in enumerate(cluster_posts, 1):
            posts_text += f"Post {i}:\n"
            posts_text += f"Title: {post['title']}\n"
            posts_text += f"Content: {post['body']}\n"
            posts_text += f"Upvotes: {post['score']}\n"
            posts_text += f"Link: {post['link']}\n\n"
        
        messages = [
            {
                "role": "system",
                "content": "You are a sharp, concise headline writer. Given a cluster of Reddit posts, identify the most interesting, unusual, or newsworthy event. Generate a headline that is LESS THAN 12 WORDS, concrete, and self-contained, so the reader immediately knows what happened. If the posts seem unrelated, focus on the single most striking or unusual post. Make it punchy, compelling, and easy to read—avoid vague or thematic summaries. For each headline generated, also provide an extended description about 1 paragraph long that provides more context and details about the event or situation. IMPORTANT: Your response must ALWAYS include both a headline and description. Never return empty content. Format your response as: HEADLINE: [your headline here] DESCRIPTION: [your description here]"
            },
            {
                "role": "user",
                "content": f"""Cluster {cluster_id} Posts: {posts_text}

Generate a headline (under 12 words) and an extended description (1 paragraph) for this cluster."""
            },
        ]

        try:
            response = await create_openai_completion(
                messages=messages,
                model=GPT5Deployment.GPT_5_NANO,
                reasoning_effort=ReasoningEffort.MINIMAL,
                max_completion_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse headline and description from response using the new format
            headline = ""
            description = ""
            
            # Look for HEADLINE: and DESCRIPTION: markers
            if "HEADLINE:" in response_text and "DESCRIPTION:" in response_text:
                parts = response_text.split("DESCRIPTION:")
                if len(parts) >= 2:
                    headline_part = parts[0].replace("HEADLINE:", "").strip()
                    description_part = parts[1].strip()
                    
                    headline = headline_part
                    description = description_part
            
            # Fallback parsing if format is different
            if not headline or not description:
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Headline:') or line.startswith('HEADLINE:'):
                        headline = line.split(':', 1)[1].strip()
                    elif line.startswith('Description:') or line.startswith('DESCRIPTION:'):
                        description = line.split(':', 1)[1].strip()
                    elif not headline and line and len(line.split()) <= 12:
                        headline = line
                    elif not description and line and len(line.split()) > 12:
                        description = line
            
            # Final fallback - use first line as headline, rest as description
            if not headline or not description:
                lines = [l.strip() for l in response_text.split('\n') if l.strip()]
                if lines:
                    headline = lines[0]
                    description = ' '.join(lines[1:]) if len(lines) > 1 else "No description provided"
            
            # Validate that we have non-empty content
            if not headline or not headline.strip() or len(headline.strip()) < 3:
                print(f"Cluster {cluster_id}: ❌ Empty headline detected, skipping")
                continue
                
            if not description or not description.strip() or len(description.strip()) < 10:
                print(f"Cluster {cluster_id}: ❌ Empty description detected, skipping")
                continue
            
            headlines[cluster_id] = {
                'headline': headline.strip(),
                'description': description.strip()
            }
            print(f"Cluster {cluster_id}: {headline.strip()}")
            
        except Exception as e:
            print(f"Cluster {cluster_id}: Error - {str(e)}")
            headlines[cluster_id] = {
                'headline': f"Cluster {cluster_id} - Analysis failed",
                'description': "Error generating description"
            }
    
    return headlines

async def select_most_interesting_headlines(headlines, cluster_analysis):
    """Use GPT to select the 3 most interesting headlines from all generated headlines."""
    print("\n🎯 Using GPT to select the 3 most interesting headlines...")
    
    # Filter out empty headlines first
    valid_headlines = {}
    for cluster_id, data in headlines.items():
        headline = data['headline'] if isinstance(data, dict) else data
        description = data['description'] if isinstance(data, dict) else ""
        
        # Skip empty or invalid headlines
        if not headline or not headline.strip() or len(headline.strip()) < 3:
            print(f"⚠️  Skipping Cluster {cluster_id}: Empty headline")
            continue
        if not description or not description.strip() or len(description.strip()) < 10:
            print(f"⚠️  Skipping Cluster {cluster_id}: Empty description")
            continue
            
        valid_headlines[cluster_id] = data
    
    if len(valid_headlines) < 3:
        print(f"❌ Only {len(valid_headlines)} valid headlines found, need at least 3")
        return {}
    
    # Prepare all headlines for GPT evaluation
    headlines_text = ""
    for cluster_id, data in valid_headlines.items():
        analysis = cluster_analysis[cluster_id]
        headline = data['headline'] if isinstance(data, dict) else data
        headlines_text += f"Cluster {cluster_id}: {headline}\n"
        headlines_text += f"  - Size: {analysis['size']} posts\n"
        headlines_text += f"  - Avg Score: {analysis['avg_score']:.1f}, Max Score: {analysis['max_score']}\n"
        headlines_text += f"  - Top Keywords: {', '.join(analysis['top_words'][:3])}\n\n"
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert content curator. Given a list of headlines from Reddit post clusters, select the 3 most interesting, engaging, and newsworthy headlines. Consider factors like uniqueness, viral potential, and general interest. Respond with ONLY the 3 cluster numbers (e.g., '3, 7, 1') in order of most to least interesting."
        },
        {
            "role": "user",
            "content": f"""Here are all the generated headlines:

{headlines_text}

Select the 3 most interesting headlines. Respond with ONLY the cluster numbers in order (e.g., '3, 7, 1'):"""
        }
    ]
    
    try:
        response = await create_openai_completion(
            messages=messages,
            model=GPT5Deployment.GPT_5_NANO,
            reasoning_effort=ReasoningEffort.MINIMAL,
            max_completion_tokens=50
        )
        
        selected_text = response.choices[0].message.content.strip()
        
        # Parse the response to get cluster IDs
        try:
            selected_clusters = [int(x.strip()) for x in selected_text.split(',')]
            if len(selected_clusters) != 3:
                raise ValueError("Not exactly 3 clusters selected")
            
            # Create the final selection using valid headlines
            final_headlines = {}
            for cluster_id in selected_clusters:
                if cluster_id in valid_headlines:
                    final_headlines[cluster_id] = valid_headlines[cluster_id]
            
            print(f"\n🎯 TOP 3 MOST INTERESTING HEADLINES:")
            for i, cluster_id in enumerate(selected_clusters, 1):
                if cluster_id in valid_headlines:
                    data = valid_headlines[cluster_id]
                    headline = data['headline'] if isinstance(data, dict) else data
                    description = data['description'] if isinstance(data, dict) else ""
                    print(f"{i}. Cluster {cluster_id}: {headline}")
                    if description:
                        print(f"   {description}")
                    print()
            
            return final_headlines
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing GPT response: {e}")
            print(f"GPT response: {selected_text}")
            # Fallback: return first 3 headlines
            first_3 = dict(list(headlines.items())[:3])
            print("Falling back to first 3 headlines")
            return first_3
            
    except Exception as e:
        print(f"Error in GPT selection: {str(e)}")
        # Fallback: return first 3 headlines
        first_3 = dict(list(headlines.items())[:3])
        print("Falling back to first 3 headlines")
        return first_3

def save_headlines_to_file(headlines, cluster_analysis, filename="cluster_headlines.txt"):
    """Save headlines to a text file with cluster details."""
    with open(filename, 'w') as f:
        f.write("UCSD REDDIT - TOP 3 MOST INTERESTING HEADLINES\n")
        f.write("="*50 + "\n\n")
        
        for cluster_id, data in headlines.items():
            analysis = cluster_analysis[cluster_id]
            headline = data['headline'] if isinstance(data, dict) else data
            description = data['description'] if isinstance(data, dict) else ""
            
            f.write(f"CLUSTER {cluster_id} ({analysis['size']} posts)\n")
            f.write(f"Headline: {headline}\n")
            if description:
                f.write(f"Description: {description}\n")
            f.write(f"Top Keywords: {', '.join(analysis['top_words'][:5])}\n")
            f.write(f"Avg Score: {analysis['avg_score']:.1f}, Max Score: {analysis['max_score']}\n\n")
            
            # Add top posts for reference
            f.write("Top 10 Representative Posts:\n")
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                f.write(f"  {idx+1}. [{post['score']} upvotes] {post['title']}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"💾 Headlines saved to {filename}")

async def store_headlines_in_supabase(headlines, cluster_analysis):
    """Store only the 3 selected headlines in Supabase tables."""
    try:
        supabase = get_supabase_client()
        print("\n💾 Storing top 3 headlines in Supabase...")
        
        # Prepare data for the 3 selected headlines only
        subreddit_data = []
        headline_data = []
        
        for cluster_id, data in headlines.items():
            headline = data['headline'] if isinstance(data, dict) else data
            description = data['description'] if isinstance(data, dict) else ""
            
            # Prepare headline_info entry first (will generate UUID)
            headline_entry = {
                'headline': headline,
                'description': description
            }
            headline_data.append(headline_entry)
        
        # Insert into headline_info table first to get UUIDs
        headline_result = supabase.table('headline_info').insert(headline_data).execute()
        
        if headline_result.data:
            print(f"✅ Inserted {len(headline_result.data)} entries into headline_info table")
            
            # Now insert into subreddit_to_headline_id table with the generated UUIDs
            for i, headline_info in enumerate(headline_result.data):
                subreddit_entry = {
                    'subreddit': 'UCSD',
                    'headline_id': headline_info['headline_id']  # UUID from headline_info table
                }
                subreddit_data.append(subreddit_entry)
            
            subreddit_result = supabase.table('subreddit_to_headline_id').insert(subreddit_data).execute()
            
            if subreddit_result.data:
                print(f"✅ Inserted {len(subreddit_result.data)} entries into subreddit_to_headline_id table")
                print("📊 Top 3 headlines successfully stored in Supabase!")
            else:
                print("❌ Failed to insert into subreddit_to_headline_id table")
        else:
            print("❌ Failed to insert into headline_info table")
            
    except Exception as e:
        print(f"❌ Error storing in Supabase: {str(e)}")
        print("Make sure you have SUPABASE_ANON_KEY in your .env file")

def print_cluster_summary(cluster_analysis):
    """Print a summary of all clusters with topic identification."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS - TOPIC IDENTIFICATION")
    print("="*80)
    
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\n🏷️  Cluster {cluster_id} ({analysis['size']} posts) - Discovered Topic")
        print(f"   Top keywords: {', '.join(analysis['top_words'][:5])}")
        print(f"   Score stats: avg={analysis['avg_score']:.1f}, max={analysis['max_score']}")
        print("   Top 10 most representative posts in this cluster:")
        
        # For first 3 clusters, show full posts
        if cluster_id < 3:
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                score = post['score']
                title = post['title']
                body = post['body']
                print(f"\n     {idx+1}. [{score} upvotes] {title}")
                print(f"        Body: {body}")
        else:
            # For other clusters, show truncated titles
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                title = post['title'][:80] + "..." if len(str(post['title'])) > 80 else post['title']
                score = post['score']
                print(f"     {idx+1}. [{score} upvotes] {title}")

async def main():
    """Main function to run the clustering pipeline."""
    print("Starting post clustering analysis...")

    # Load data
    df = load_posts('UCSD.tsv')
    
    # Generate embeddings
    embeddings = await generate_embeddings(df['combined_text'].tolist())
    
    # Find optimal number of clusters using multiple methods
    optimal_k, results = find_optimal_clusters(embeddings)
    
    # Cluster posts
    cluster_labels, best_algorithm = cluster_posts(embeddings, optimal_k)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df, cluster_labels, embeddings)
    
    # Skip verbose cluster summary for cleaner output
    # print_cluster_summary(cluster_analysis)
    
    # Generate headlines with GPT
    all_headlines = await generate_cluster_headlines(cluster_analysis)
    
    # Use GPT to select the 3 most interesting headlines
    best_headlines = await select_most_interesting_headlines(all_headlines, cluster_analysis)
    
    # Save best headlines to file
    save_headlines_to_file(best_headlines, cluster_analysis)
    
    # Store headlines in Supabase
    await store_headlines_in_supabase(best_headlines, cluster_analysis)
    
    # Save results
    df_with_clusters = df.copy()
    df_with_clusters.to_csv('posts_with_clusters.tsv', sep='\t', index=False)
    # Results saved silently
    
    # Save cluster analysis
    with open('cluster_analysis.txt', 'w') as f:
        f.write("CLUSTER ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Optimal number of clusters: {optimal_k}\n")
        f.write("Determined using multiple methods: Silhouette, Calinski-Harabasz, Davies-Bouldin, and Elbow method.\n\n")
        
        for cluster_id, analysis in cluster_analysis.items():
            f.write(f"Cluster {cluster_id} ({analysis['size']} posts) - Discovered Topic\n")
            f.write(f"  Top keywords: {', '.join(analysis['top_words'])}\n")
            f.write(f"  Score stats: avg={analysis['avg_score']:.1f}, max={analysis['max_score']}\n")
            f.write("  Top 10 most representative posts in this cluster:\n")
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                title = post['title'][:100] + "..." if len(str(post['title'])) > 100 else post['title']
                score = post['score']
                f.write(f"    {idx+1}. [{score} upvotes] {title}\n")
            f.write("\n")
    
    # Analysis complete silently

if __name__ == "__main__":
    asyncio.run(main())