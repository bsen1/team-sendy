#!/usr/bin/env python3
"""
College Insights Analyzer
Scrapes college subreddit data and provides comprehensive insights about the college
"""

import asyncio
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Import our existing modules
from reddit_scraper import RedditScraper, RedditPost
from unwrap_openai import create_openai_completion, GPT5Deployment, ReasoningEffort

# Load environment variables
load_dotenv()


@dataclass
class CollegeInsight:
    """Comprehensive college insight analysis"""
    college_name: str
    subreddit: str
    analysis_date: datetime
    total_posts_analyzed: int
    top_keywords: List[tuple]
    keyword_analysis: Dict[str, Any]
    college_summary: str
    key_concerns: List[str]
    positive_topics: List[str]
    recommendations: List[str]
    sentiment_overview: str


class CollegeInsightsAnalyzer:
    """AI-powered college insights analyzer using Reddit data"""
    
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, reddit_user_agent: str):
        """Initialize with Reddit API credentials"""
        self.reddit_scraper = RedditScraper(reddit_client_id, reddit_client_secret, reddit_user_agent)
    
    async def analyze_post_keyword(self, post: RedditPost) -> Dict[str, Any]:
        """
        Analyze a Reddit post and identify the main keyword/topic
        
        Args:
            post: RedditPost object to analyze
            
        Returns:
            Dictionary with keyword analysis
        """
        # Prepare the content for analysis
        content = f"Title: {post.title}\nContent: {post.content}"
        
        # Truncate if too long (keep under 2000 chars for efficiency)
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert at analyzing Reddit posts from college subreddits and identifying the main topic/keyword. 
                Your task is to summarize each post into a single, descriptive word that captures the essence of what the post is about.
                
                Focus on college-specific topics:
                - Academic topics (courses, majors, professors, assignments, grades, research)
                - Campus life (dorms, dining, activities, clubs, study spaces, facilities)
                - Social issues (relationships, mental health, stress, roommate problems, social life)
                - Career topics (internships, jobs, graduate school, resume building, networking)
                - Administrative issues (financial aid, registration, policies, scholarships, bureaucracy)
                - Campus culture (traditions, events, student life, diversity)
                
                Return your analysis in this exact JSON format:
                {
                    "keyword": "single_word_describing_topic",
                    "confidence": 0.95,
                    "reasoning": "brief explanation of why this keyword fits",
                    "sentiment": "positive/negative/neutral",
                    "urgency": "high/medium/low"
                }"""
            },
            {
                "role": "user",
                "content": f"Analyze this Reddit post from r/{post.subreddit}:\n\n{content}"
            }
        ]
        
        try:
            response = await create_openai_completion(
                messages=messages,
                model=GPT5Deployment.GPT_5_NANO,
                reasoning_effort=ReasoningEffort.MINIMAL,
                max_completion_tokens=200
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Fallback: create a simple analysis
                json_text = f'{{"keyword": "general", "confidence": 0.5, "reasoning": "Could not parse AI response", "sentiment": "neutral", "urgency": "medium"}}'
            
            analysis_data = json.loads(json_text)
            
            return {
                'post_id': post.post_id,
                'title': post.title,
                'content': post.content[:500],
                'subreddit': post.subreddit,
                'keyword': analysis_data.get("keyword", "unknown"),
                'confidence': analysis_data.get("confidence", 0.5),
                'reasoning': analysis_data.get("reasoning", "No reasoning provided"),
                'sentiment': analysis_data.get("sentiment", "neutral"),
                'urgency': analysis_data.get("urgency", "medium"),
                'score': post.score,
                'num_comments': post.num_comments,
                'timestamp': post.timestamp.isoformat()
            }
            
        except Exception as e:
            # Fallback analysis if AI fails
            return {
                'post_id': post.post_id,
                'title': post.title,
                'content': post.content[:500],
                'subreddit': post.subreddit,
                'keyword': "error",
                'confidence': 0.0,
                'reasoning': f"Analysis failed: {str(e)}",
                'sentiment': "neutral",
                'urgency': "low",
                'score': post.score,
                'num_comments': post.num_comments,
                'timestamp': post.timestamp.isoformat()
            }
    
    async def generate_college_insights(self, analyses: List[Dict[str, Any]], college_name: str) -> str:
        """
        Generate comprehensive college insights from keyword analyses
        
        Args:
            analyses: List of post analyses
            college_name: Name of the college
            
        Returns:
            Comprehensive insights about the college
        """
        # Prepare data for analysis
        keyword_counts = {}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        urgency_counts = {"high": 0, "medium": 0, "low": 0}
        
        for analysis in analyses:
            keyword = analysis['keyword'].lower()
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            sentiment_counts[analysis['sentiment']] += 1
            urgency_counts[analysis['urgency']] += 1
        
        # Sort keywords by frequency
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Prepare context for AI analysis
        context = f"""
        College: {college_name}
        Total Posts Analyzed: {len(analyses)}
        
        Top Keywords and Frequencies:
        {chr(10).join([f"- {keyword}: {count} posts" for keyword, count in top_keywords])}
        
        Sentiment Distribution:
        - Positive: {sentiment_counts['positive']} posts
        - Negative: {sentiment_counts['negative']} posts  
        - Neutral: {sentiment_counts['neutral']} posts
        
        Urgency Distribution:
        - High: {urgency_counts['high']} posts
        - Medium: {urgency_counts['medium']} posts
        - Low: {urgency_counts['low']} posts
        
        Sample Post Topics:
        {chr(10).join([f"- {analysis['title'][:80]}..." for analysis in analyses[:10]])}
        """
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert college administrator and student affairs professional with deep experience analyzing student sentiment and campus issues. 
                
                Based on Reddit post analysis from a college's subreddit, provide comprehensive insights about the college including:
                
                1. Overall College Summary (2-3 sentences about the general state of the college)
                2. Key Student Concerns (top 5-7 issues students are discussing)
                3. Positive Topics (what students are happy about)
                4. Recommendations (actionable suggestions for college administration)
                5. Sentiment Overview (overall student satisfaction and mood)
                
                Be specific, actionable, and focus on insights that would help college administrators understand and address student needs.
                
                Return your analysis in this exact JSON format:
                {
                    "college_summary": "2-3 sentence overview of the college's current state",
                    "key_concerns": ["concern1", "concern2", "concern3", "concern4", "concern5"],
                    "positive_topics": ["positive1", "positive2", "positive3"],
                    "recommendations": ["recommendation1", "recommendation2", "recommendation3", "recommendation4"],
                    "sentiment_overview": "overall assessment of student satisfaction and mood"
                }"""
            },
            {
                "role": "user",
                "content": f"Analyze this college based on Reddit subreddit data:\n\n{context}"
            }
        ]
        
        try:
            response = await create_openai_completion(
                messages=messages,
                model=GPT5Deployment.GPT_5_MINI,
                reasoning_effort=ReasoningEffort.LOW,
                max_completion_tokens=1000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Fallback: create a simple analysis
                json_text = f'{{"college_summary": "Analysis incomplete due to parsing error", "key_concerns": ["Data parsing issue"], "positive_topics": ["Unable to determine"], "recommendations": ["Review data processing"], "sentiment_overview": "Unable to assess"}}'
            
            return json.loads(json_text)
            
        except Exception as e:
            # Fallback analysis if AI fails
            return {
                "college_summary": f"Analysis failed: {str(e)}",
                "key_concerns": ["Technical error in analysis"],
                "positive_topics": ["Unable to determine"],
                "recommendations": ["Review system functionality"],
                "sentiment_overview": "Unable to assess due to technical issues"
            }
    
    async def analyze_college(self, college_subreddit: str, college_name: str, limit: int = 25, sort_method: str = "hot") -> CollegeInsight:
        """
        Complete college analysis from subreddit data
        
        Args:
            college_subreddit: Subreddit name (without r/)
            college_name: Display name of the college
            limit: Number of posts to analyze
            
        Returns:
            CollegeInsight object with comprehensive analysis
        """
        print(f"🎓 Analyzing {college_name} (r/{college_subreddit})...")
        print(f"🔍 Scraping top {limit} posts sorted by '{sort_method}'...")
        
        # Scrape posts
        posts = self.reddit_scraper.scrape_subreddit(college_subreddit, sort_method, limit)
        
        if not posts:
            print(f"❌ No posts found for r/{college_subreddit}")
            return None
        
        print(f"📊 Found {len(posts)} posts. Analyzing keywords...")
        
        # Analyze each post
        analyses = []
        for i, post in enumerate(posts, 1):
            print(f"  Analyzing post {i}/{len(posts)}: {post.title[:50]}...")
            analysis = await self.analyze_post_keyword(post)
            analyses.append(analysis)
        
        print(f"🧠 Generating college insights...")
        
        # Generate comprehensive insights
        insights_data = await self.generate_college_insights(analyses, college_name)
        
        # Count keyword frequencies for summary
        keyword_counts = {}
        for analysis in analyses:
            keyword = analysis['keyword'].lower()
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return CollegeInsight(
            college_name=college_name,
            subreddit=college_subreddit,
            analysis_date=datetime.now(),
            total_posts_analyzed=len(analyses),
            top_keywords=top_keywords,
            keyword_analysis=analyses,
            college_summary=insights_data.get("college_summary", "Analysis incomplete"),
            key_concerns=insights_data.get("key_concerns", []),
            positive_topics=insights_data.get("positive_topics", []),
            recommendations=insights_data.get("recommendations", []),
            sentiment_overview=insights_data.get("sentiment_overview", "Unable to assess")
        )


async def main():
    """Main function to demonstrate college insights analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="College Insights Analyzer")
    parser.add_argument("subreddit", help="College subreddit name (without r/)")
    parser.add_argument("-n", "--name", help="College name (default: subreddit name)")
    parser.add_argument("-l", "--limit", type=int, default=25,
                       help="Number of posts to analyze (default: 25)")
    parser.add_argument("-s", "--sort", default="hot",
                       choices=["hot", "new", "rising", "top", "controversial"],
                       help="Sort method: hot=trending, new=recent, rising=popular, top=highest scoring, controversial=mixed votes (default: hot)")
    parser.add_argument("-o", "--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Load credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    
    if not all([client_id, client_secret, user_agent]):
        print("❌ Error: Missing Reddit API credentials")
        print("Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT environment variables")
        print("\nTo get Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Choose 'script' as the app type")
        print("4. Fill in the details and note the Client ID and Secret")
        return
    
    # Initialize analyzer
    analyzer = CollegeInsightsAnalyzer(client_id, client_secret, user_agent)
    
    # Analyze college
    college_name = args.name or args.subreddit.replace("_", " ").title()
    insight = await analyzer.analyze_college(args.subreddit, college_name, args.limit, args.sort)
    
    if not insight:
        print("❌ Analysis failed")
        return
    
    # Display results
    print(f"\n{'='*80}")
    print(f"🎓 COLLEGE INSIGHTS: {insight.college_name.upper()}")
    print(f"📊 Subreddit: r/{insight.subreddit}")
    print(f"📅 Analysis Date: {insight.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 Posts Analyzed: {insight.total_posts_analyzed}")
    print(f"{'='*80}")
    
    print(f"\n📋 COLLEGE SUMMARY:")
    print(f"{insight.college_summary}")
    
    print(f"\n🔥 TOP KEYWORDS:")
    for keyword, count in insight.top_keywords:
        print(f"  {keyword}: {count} posts")
    
    print(f"\n⚠️  KEY STUDENT CONCERNS:")
    for i, concern in enumerate(insight.key_concerns, 1):
        print(f"  {i}. {concern}")
    
    print(f"\n✅ POSITIVE TOPICS:")
    for i, topic in enumerate(insight.positive_topics, 1):
        print(f"  {i}. {topic}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(insight.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n😊 SENTIMENT OVERVIEW:")
    print(f"{insight.sentiment_overview}")
    
    # Save to file if requested
    if args.output:
        output_data = {
            'college_name': insight.college_name,
            'subreddit': insight.subreddit,
            'analysis_date': insight.analysis_date.isoformat(),
            'total_posts_analyzed': insight.total_posts_analyzed,
            'top_keywords': insight.top_keywords,
            'college_summary': insight.college_summary,
            'key_concerns': insight.key_concerns,
            'positive_topics': insight.positive_topics,
            'recommendations': insight.recommendations,
            'sentiment_overview': insight.sentiment_overview,
            'detailed_analyses': insight.keyword_analysis
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())