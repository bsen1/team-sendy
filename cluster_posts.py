#!/usr/bin/env python3
"""
Post Clustering Script using HuggingFace Sentence Transformers
This script loads posts from posts.tsv, generates embeddings, and clusters them to identify topics.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

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

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate sentence embeddings using HuggingFace sentence-transformers."""
    print(f"Generating embeddings using {model_name}...")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def test_clustering_algorithms(embeddings, optimal_k):
    """Test different clustering algorithms and return the best one."""
    print(f"\nTesting different clustering algorithms with k={optimal_k}...")
    
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
                print(f"  {name}: silhouette = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_algorithm = name
            else:
                print(f"  {name}: failed (only 1 cluster)")
                results[name] = {'labels': cluster_labels, 'score': -1}
                
        except Exception as e:
            print(f"  {name}: failed - {str(e)}")
            results[name] = {'labels': None, 'score': -1}
    
    print(f"\nBest algorithm: {best_algorithm} (silhouette = {best_score:.4f})")
    return results[best_algorithm]['labels'], best_algorithm

def find_optimal_clusters(embeddings, max_k=None):
    """Find optimal number of clusters using multiple methods."""
    print("Finding optimal number of clusters using multiple methods...")
    
    n_samples = len(embeddings)
    
    # More fine-grained range for detailed topic clustering
    if max_k is None:
        # Focus on 5-25 clusters for fine-grained topic separation
        min_k = 5
        max_k = min(25, n_samples // 30)  # More clusters for detailed topics
    else:
        min_k = 5

    print(f"Testing k from {min_k} to {max_k} for fine-grained topic clustering")
    
    k_range = range(min_k, max_k + 1)
    
    # Store results for different methods
    results = {
        'k': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
        'inertia': []
    }
    
    print("Evaluating different numbers of clusters...")
    for k in k_range:
        print(f"  Testing k={k}...", end=' ')
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        inertia = kmeans.inertia_
        
        results['k'].append(k)
        results['silhouette'].append(silhouette_avg)
        results['calinski_harabasz'].append(calinski_harabasz)
        results['davies_bouldin'].append(davies_bouldin)
        results['inertia'].append(inertia)
        
        print(f"silhouette={silhouette_avg:.3f}")
    
    # Find optimal k using different methods
    silhouette_optimal = k_range[np.argmax(results['silhouette'])]
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
    
    print(f"  Consensus optimal k: {consensus_k}")
    
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
        
        # Get the 5 posts with highest combined score
        best_indices = np.argsort(combined_score)[::-1][:5]
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

def print_cluster_summary(cluster_analysis):
    """Print a summary of all clusters with topic identification."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS - TOPIC IDENTIFICATION")
    print("="*80)
    
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\n🏷️  Cluster {cluster_id} ({analysis['size']} posts) - Discovered Topic")
        print(f"   Top keywords: {', '.join(analysis['top_words'][:5])}")
        print(f"   Score stats: avg={analysis['avg_score']:.1f}, max={analysis['max_score']}")
        print("   Top 5 most representative posts in this cluster:")
        
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

def main():
    """Main function to run the clustering pipeline."""
    print("Starting post clustering analysis...")
    print("This will automatically determine the optimal number of clusters using multiple methods.")
    
    # Load data
    df = load_posts('ucla.tsv')
    
    # Generate embeddings
    embeddings = generate_embeddings(df['combined_text'].tolist())
    
    # Find optimal number of clusters using multiple methods
    optimal_k, results = find_optimal_clusters(embeddings)
    
    # Cluster posts
    cluster_labels, best_algorithm = cluster_posts(embeddings, optimal_k)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df, cluster_labels, embeddings)
    
    # Print summary
    print_cluster_summary(cluster_analysis)
    
    # Save results
    df_with_clusters = df.copy()
    df_with_clusters.to_csv('posts_with_clusters.tsv', sep='\t', index=False)
    print(f"\nResults saved to 'posts_with_clusters.tsv'")
    
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
            f.write("  Top 5 most representative posts in this cluster:\n")
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                title = post['title'][:100] + "..." if len(str(post['title'])) > 100 else post['title']
                score = post['score']
                f.write(f"    {idx+1}. [{score} upvotes] {title}\n")
            f.write("\n")
    
    print(f"\nAnalysis complete! Used {best_algorithm} clustering algorithm.")

if __name__ == "__main__":
    main()