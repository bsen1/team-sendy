# Reddit Scraper

A clean, focused Reddit scraper built on PRAW (Python Reddit API Wrapper) that returns structured data for analysis.

## Features

- 🎯 **Subreddit scraping** with flexible sorting options
- 📊 **Structured data** with RedditPost objects
- 🔍 **Keyword filtering** for targeted content
- ⚡ **Clean API** for programmatic usage
- 🚀 **Command line interface** for quick testing

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Set Up Reddit API Credentials

Create a `.env` file in the project root:

```bash
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=your_user_agent_here
```

Get credentials from https://www.reddit.com/prefs/apps

### 3. Run the Scraper

```bash
# Activate virtual environment
source venv/bin/activate

# Basic scraping
python reddit_scraper.py funny -s hot -l 10

# With keyword filtering
python reddit_scraper.py technology -s hot -l 10 -k AI machine learning
```

## Usage Examples

### Basic Scraping

```bash
# Get hot posts from a subreddit
python reddit_scraper.py funny -s hot -l 10

# Get new posts
python reddit_scraper.py technology -s new -l 5

# Get top posts
python reddit_scraper.py programming -s top -l 20
```

### Sort Options

- `hot` - Most popular posts (default)
- `new` - Most recent posts
- `rising` - Posts gaining traction
- `top` - Highest scoring posts
- `controversial` - Most controversial posts

### Keyword Filtering

```bash
# Filter by keywords (any match)
python reddit_scraper.py technology -s hot -l 10 -k AI artificial intelligence
python reddit_scraper.py funny -s hot -l 10 -k meme joke
```

## Programmatic Usage

```python
from reddit_scraper import RedditScraper, RedditPost
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize scraper
scraper = RedditScraper(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Scrape posts
posts = scraper.scrape_subreddit("funny", "hot", 10, ["meme", "joke"])

# Access structured data
for post in posts:
    print(f"Title: {post.title}")
    print(f"Score: {post.score}")
    print(f"Author: {post.author}")
    print(f"Timestamp: {post.timestamp}")
    print(f"Content: {post.content}")
    print(f"URL: {post.url}")
    print("-" * 40)
```

## RedditPost Object

Each scraped post is returned as a structured `RedditPost` object with:

- `title` - Post title
- `content` - Post text content
- `author` - Post author
- `score` - Upvotes minus downvotes
- `num_comments` - Number of comments
- `subreddit` - Subreddit name
- `url` - Direct link to post
- `timestamp` - When posted
- `upvote_ratio` - Ratio of upvotes to total votes
- `post_id` - Unique Reddit post ID

### Convert to Dictionary

```python
# Convert to dictionary for JSON serialization
post_dict = post.to_dict()
```

## Command Line Options

- `subreddit` - Subreddit name (without r/)
- `-s, --sort` - Sort method (hot, new, rising, top, controversial)
- `-l, --limit` - Number of posts to scrape (default: 10)
- `-k, --keywords` - Keywords to filter by (any match)

## Output Format

Results are displayed in the terminal with:

- Post number and title
- Author and score
- Number of comments
- Subreddit and date
- Upvote ratio and URL
- Content preview (first 200 characters)

## Use Cases

### College Complaints Analysis

```bash
# Scrape college subreddits for complaints
python reddit_scraper.py UCSantaBarbara -s hot -l 20 -k complaint issue problem
python reddit_scraper.py MIT -s hot -l 20 -k complaint issue problem
```

### Technology Research

```bash
# Get latest tech discussions
python reddit_scraper.py technology -s new -l 10
python reddit_scraper.py programming -s hot -l 10 -k python javascript
```

### Community Insights

```bash
# Analyze community sentiment
python reddit_scraper.py startups -s hot -l 20 -k feedback experience
```

## Requirements

- Python 3.7+
- PRAW (Python Reddit API Wrapper)
- python-dotenv
- Reddit API credentials

## Rate Limiting

Reddit has API rate limits. The scraper respects these limits automatically. For heavy usage:

- Use reasonable limits (10-50 posts per request)
- Wait between large scraping sessions
- Follow Reddit's API guidelines

## License

MIT License - feel free to use for your projects!
