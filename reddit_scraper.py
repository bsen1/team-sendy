#!/usr/bin/env python3
"""
Clean Reddit Scraper - Core Function Only
"""

import praw
import os
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class RedditPost:
    """Structured Reddit post data"""
    title: str
    content: str
    author: str
    score: int
    num_comments: int
    subreddit: str
    url: str
    timestamp: datetime
    upvote_ratio: float
    post_id: str
    
    def to_dict(self):
        """Convert to dictionary for easy serialization"""
        return {
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'score': self.score,
            'num_comments': self.num_comments,
            'subreddit': self.subreddit,
            'url': self.url,
            'timestamp': self.timestamp.isoformat(),
            'upvote_ratio': self.upvote_ratio,
            'post_id': self.post_id
        }


class RedditScraper:
    """Clean Reddit scraper with core functionality"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def scrape_subreddit(self, subreddit_name: str, sort_by: str = "hot", 
                        limit: int = 10, keywords: Optional[List[str]] = None) -> List[RedditPost]:
        """
        Scrape posts from a subreddit with optional keyword filtering
        
        Args:
            subreddit_name: Name of subreddit (without r/)
            sort_by: Sort method (hot, new, rising, top, controversial)
            limit: Number of posts to scrape
            keywords: Optional list of keywords to filter by (any match)
            
        Returns:
            List of RedditPost objects
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sort method
            if sort_by == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort_by == "new":
                posts = subreddit.new(limit=limit)
            elif sort_by == "rising":
                posts = subreddit.rising(limit=limit)
            elif sort_by == "top":
                posts = subreddit.top(limit=limit)
            elif sort_by == "controversial":
                posts = subreddit.controversial(limit=limit)
            else:
                raise ValueError(f"Invalid sort method: {sort_by}")
            
            results = []
            for post in posts:
                # Check keyword filter if provided
                if keywords:
                    text_to_search = f"{post.title} {post.selftext}".lower()
                    if not any(keyword.lower() in text_to_search for keyword in keywords):
                        continue
                
                # Create RedditPost object
                reddit_post = RedditPost(
                    title=post.title,
                    content=post.selftext or "",
                    author=str(post.author) if post.author else "[deleted]",
                    score=post.score,
                    num_comments=post.num_comments,
                    subreddit=subreddit_name,
                    url=f"https://reddit.com{post.permalink}",
                    timestamp=datetime.fromtimestamp(post.created_utc),
                    upvote_ratio=post.upvote_ratio,
                    post_id=post.id
                )
                
                results.append(reddit_post)
            
            return results
            
        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {e}")
            return []


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Reddit Scraper")
    parser.add_argument("subreddit", help="Subreddit name (without r/)")
    parser.add_argument("-s", "--sort", default="hot", 
                       choices=["hot", "new", "rising", "top", "controversial"],
                       help="Sort method (default: hot)")
    parser.add_argument("-l", "--limit", type=int, default=10,
                       help="Number of posts to scrape (default: 10)")
    parser.add_argument("-k", "--keywords", nargs="+",
                       help="Keywords to filter by (any match)")
    
    args = parser.parse_args()
    
    # Load credentials from environment
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    
    if not all([client_id, client_secret, user_agent]):
        print("Error: Missing Reddit API credentials in environment variables")
        print("Required: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return
    
    # Initialize scraper
    scraper = RedditScraper(client_id, client_secret, user_agent)
    
    # Scrape posts
    posts = scraper.scrape_subreddit(args.subreddit, args.sort, args.limit, args.keywords)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"R/{args.subreddit.upper()} POSTS (SORTED BY {args.sort.upper()}) ({len(posts)} posts)")
    print(f"{'='*80}")
    
    for i, post in enumerate(posts, 1):
        print(f"\nPost #{i}")
        print(f"Title: {post.title}")
        print(f"Author: {post.author}")
        print(f"Score: {post.score} | Comments: {post.num_comments}")
        print(f"Subreddit: r/{post.subreddit}")
        print(f"Date: {post.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Upvote Ratio: {post.upvote_ratio:.2f}")
        print(f"URL: {post.url}")
        if post.content:
            print(f"Content: {post.content[:200]}{'...' if len(post.content) > 200 else ''}")
        print("-" * 80)


if __name__ == "__main__":
    main()