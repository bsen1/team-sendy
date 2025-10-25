#!/usr/bin/env python3
"""
Post Clustering Script using HuggingFace Sentence Transformers
This script loads posts from posts.tsv, generates embeddings, and clusters them to identify topics.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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

def find_optimal_clusters(embeddings, max_k=None):
    """Find optimal number of clusters using multiple methods."""
    print("Finding optimal number of clusters using multiple methods...")
    
    n_samples = len(embeddings)
    
    # Dynamic range calculation
    if max_k is None:
        # Rule of thumb: sqrt(n_samples) to n_samples/10, but with reasonable bounds
        min_k = max(2, int(np.sqrt(n_samples) // 2))
        max_k = min(n_samples // 10, 50)  # Cap at 50 for performance
    else:
        min_k = 2
    
    print(f"Testing k from {min_k} to {max_k}")
    
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
    
    # Consensus approach: take the most common optimal k
    optimal_ks = [silhouette_optimal, calinski_optimal, davies_optimal, elbow_optimal]
    k_counts = Counter(optimal_ks)
    consensus_k = k_counts.most_common(1)[0][0]
    
    print(f"  Consensus optimal k: {consensus_k}")
    
    return consensus_k, results

def cluster_posts(embeddings, n_clusters):
    """Cluster posts using K-means."""
    print(f"Clustering posts into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, kmeans

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
        
        # Filter out common words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 
            'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'after',
            'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'here', 'just',
            'into', 'over', 'think', 'know', 'like', 'make', 'more', 'than', 'then', 'them',
            'these', 'through', 'what', 'your', 'about', 'all', 'and', 'are', 'but', 'for',
            'had', 'has', 'her', 'him', 'his', 'how', 'its', 'may', 'not', 'now', 'one',
            'our', 'out', 'she', 'the', 'was', 'you', 'can', 'get', 'use', 'way', 'year',
            'work', 'good', 'new', 'see', 'him', 'two', 'who', 'boy', 'did', 'its', 'let',
            'put', 'say', 'she', 'too', 'use', 'want', 'any', 'day', 'may', 'old', 'see',
            'try', 'ask', 'came', 'end', 'why', 'back', 'came', 'end', 'why', 'back',
            'ucsb', 'student', 'students', 'campus', 'school', 'university', 'college'
        }
        
        meaningful_words = [word for word in words if word not in stop_words]
        word_freq = Counter(meaningful_words)
        top_words = [word for word, freq in word_freq.most_common(10)]
        
        # Get sample posts from this cluster
        sample_posts = cluster_posts[['title', 'body']].head(5)
        
        # Identify topic based on keywords
        topic_keywords = {
            'housing': ['housing', 'apartment', 'dorm', 'room', 'rent', 'lease', 'sublease', 'roommate'],
            'parking': ['parking', 'car', 'vehicle', 'spot', 'garage', 'lot'],
            'academic': ['class', 'course', 'professor', 'exam', 'midterm', 'final', 'grade', 'gpa', 'major'],
            'financial': ['financial', 'aid', 'money', 'cost', 'expensive', 'cheap', 'budget', 'tuition'],
            'food': ['food', 'dining', 'restaurant', 'meal', 'eat', 'cafe', 'cafeteria'],
            'social': ['party', 'event', 'social', 'friend', 'meet', 'club', 'organization'],
            'transportation': ['bus', 'bike', 'walk', 'transport', 'commute', 'drive']
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for word in top_words if word in keywords)
            topic_scores[topic] = score
        
        # Determine primary topic
        primary_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'general'
        if topic_scores[primary_topic] == 0:
            primary_topic = 'general'
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_posts),
            'top_words': top_words,
            'sample_posts': sample_posts,
            'primary_topic': primary_topic,
            'topic_scores': topic_scores
        }
    
    return cluster_analysis

def print_cluster_summary(cluster_analysis):
    """Print a summary of all clusters with topic identification."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS - TOPIC IDENTIFICATION")
    print("="*80)
    
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\n🏷️  Cluster {cluster_id} ({analysis['size']} posts) - Topic: {analysis['primary_topic'].upper()}")
        print(f"   Top keywords: {', '.join(analysis['top_words'][:5])}")
        print(f"   Topic scores: {analysis['topic_scores']}")
        print("   Sample posts:")
        for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
            title = post['title'][:80] + "..." if len(str(post['title'])) > 80 else post['title']
            print(f"     {idx+1}. {title}")

def main():
    """Main function to run the clustering pipeline."""
    print("Starting post clustering analysis...")
    print("This will automatically determine the optimal number of clusters using multiple methods.")
    
    # Load data
    df = load_posts('posts.tsv')
    
    # Generate embeddings
    embeddings = generate_embeddings(df['combined_text'].tolist())
    
    # Find optimal number of clusters using multiple methods
    optimal_k, results = find_optimal_clusters(embeddings)
    
    # Cluster posts
    cluster_labels, kmeans = cluster_posts(embeddings, optimal_k)
    
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
            f.write(f"Cluster {cluster_id} ({analysis['size']} posts) - Topic: {analysis['primary_topic'].upper()}\n")
            f.write(f"  Top keywords: {', '.join(analysis['top_words'])}\n")
            f.write(f"  Topic scores: {analysis['topic_scores']}\n")
            f.write("  Sample posts:\n")
            for idx, (_, post) in enumerate(analysis['sample_posts'].iterrows()):
                title = post['title'][:100] + "..." if len(str(post['title'])) > 100 else post['title']
                f.write(f"    {idx+1}. {title}\n")
            f.write("\n")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()