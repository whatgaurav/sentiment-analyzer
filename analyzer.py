"""
Driver Pain Points Analyzer for India
Scrapes Reddit for driver-related discussions and identifies key pain points
using sentiment analysis and keyword extraction.
"""

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import praw
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Download required NLTK data
def setup_nltk():
    """Download required NLTK resources."""
    resources = ['vader_lexicon', 'punkt', 'stopwords', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

setup_nltk()


@dataclass
class RedditPost:
    """Represents a Reddit post or comment."""
    text: str
    score: int
    subreddit: str
    post_type: str  # 'submission' or 'comment'
    url: str


@dataclass
class PainPoint:
    """Represents an identified pain point."""
    category: str
    description: str
    sentiment_score: float
    frequency: int
    example_quotes: list[str]
    severity: str  # 'high', 'medium', 'low'


class RedditScraper:
    """Handles Reddit API interactions."""
    
    def __init__(self):
        load_dotenv()
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'DriverPainPointsAnalyzer/1.0')
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found. Please set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in your .env file.\n"
                "Get credentials at: https://www.reddit.com/prefs/apps"
            )
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def search_driver_content(self, limit: int = 100) -> list[RedditPost]:
        """Search for driver-related content in Indian subreddits."""
        
        # Indian subreddits where driver discussions happen
        subreddits = [
            'india', 'bangalore', 'mumbai', 'delhi', 'chennai', 
            'hyderabad', 'pune', 'kolkata', 'IndianGaming',
            'CarsIndia', 'IndianBikes', 'indiasocial'
        ]
        
        # Search queries related to drivers
        search_queries = [
            'uber driver india',
            'ola driver problems',
            'cab driver india',
            'auto driver',
            'delivery driver india',
            'swiggy zomato driver',
            'truck driver india',
            'driver salary india',
            'driver issues india',
            'rapido driver',
            'gig worker driver india'
        ]
        
        posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search with various queries
                for query in search_queries:
                    try:
                        for submission in subreddit.search(query, limit=limit // len(search_queries)):
                            # Add submission text
                            text = f"{submission.title} {submission.selftext}".strip()
                            if text and self._is_driver_related(text):
                                posts.append(RedditPost(
                                    text=text,
                                    score=submission.score,
                                    subreddit=subreddit_name,
                                    post_type='submission',
                                    url=submission.url
                                ))
                            
                            # Add top comments
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments.list()[:10]:
                                if hasattr(comment, 'body') and self._is_driver_related(comment.body):
                                    posts.append(RedditPost(
                                        text=comment.body,
                                        score=comment.score,
                                        subreddit=subreddit_name,
                                        post_type='comment',
                                        url=submission.url
                                    ))
                    except Exception as e:
                        print(f"  Warning: Could not search '{query}' in r/{subreddit_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Could not access r/{subreddit_name}: {e}")
                continue
        
        print(f"Collected {len(posts)} driver-related posts/comments")
        return posts
    
    def _is_driver_related(self, text: str) -> bool:
        """Check if text is related to drivers/driving profession."""
        driver_keywords = [
            'driver', 'uber', 'ola', 'cab', 'taxi', 'auto', 'rickshaw',
            'delivery', 'swiggy', 'zomato', 'dunzo', 'rapido', 'porter',
            'truck', 'lorry', 'driving', 'ride', 'fare', 'passenger',
            'gig', 'commission', 'incentive', 'rating', 'trip'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in driver_keywords)


class SentimentAnalyzer:
    """Performs sentiment analysis on text."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using VADER and TextBlob."""
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_negative_sentences(self, text: str, threshold: float = -0.3) -> list[str]:
        """Extract sentences with negative sentiment."""
        sentences = sent_tokenize(text)
        negative_sentences = []
        
        for sentence in sentences:
            sentiment = self.sia.polarity_scores(sentence)
            if sentiment['compound'] < threshold:
                negative_sentences.append(sentence)
        
        return negative_sentences


class PainPointExtractor:
    """Extracts and categorizes pain points from driver-related content."""
    
    # Pain point categories with associated keywords
    PAIN_POINT_CATEGORIES = {
        'low_earnings': {
            'keywords': ['salary', 'earning', 'income', 'pay', 'money', 'commission', 
                        'incentive', 'fare', 'rate', 'minimum', 'wage', 'profit', 'loss',
                        'expensive', 'cost', 'fuel', 'petrol', 'diesel', 'cng'],
            'description': 'Low earnings, high costs, and unfair commission structures'
        },
        'app_platform_issues': {
            'keywords': ['app', 'crash', 'bug', 'algorithm', 'allocation', 'assign',
                        'cancel', 'penalty', 'block', 'deactivate', 'suspend', 'ban',
                        'support', 'customer care', 'helpline', 'response'],
            'description': 'Issues with ride-hailing apps, algorithms, and platform support'
        },
        'safety_concerns': {
            'keywords': ['safety', 'safe', 'danger', 'attack', 'assault', 'robbery',
                        'threat', 'harassment', 'abuse', 'violence', 'accident', 'risk',
                        'night', 'area', 'location', 'police'],
            'description': 'Safety risks, harassment, and dangerous situations'
        },
        'work_life_balance': {
            'keywords': ['hour', 'time', 'long', 'shift', 'rest', 'break', 'tired',
                        'exhausted', 'family', 'health', 'stress', 'pressure', 'sleep',
                        'overtime', 'target', 'quota'],
            'description': 'Long working hours, stress, and lack of work-life balance'
        },
        'customer_behavior': {
            'keywords': ['passenger', 'customer', 'rider', 'rude', 'cancel', 'rating',
                        'review', 'complaint', 'behavior', 'respect', 'argue', 'fight',
                        'tip', 'cash', 'payment', 'fraud'],
            'description': 'Difficult customers, unfair ratings, and payment issues'
        },
        'vehicle_maintenance': {
            'keywords': ['vehicle', 'car', 'bike', 'maintenance', 'repair', 'service',
                        'emi', 'loan', 'insurance', 'permit', 'document', 'fine',
                        'traffic', 'challan', 'police'],
            'description': 'Vehicle costs, maintenance burden, and regulatory issues'
        },
        'lack_of_benefits': {
            'keywords': ['insurance', 'health', 'medical', 'benefit', 'sick', 'leave',
                        'holiday', 'protection', 'rights', 'union', 'contract', 'job',
                        'security', 'permanent', 'gig'],
            'description': 'No job security, health benefits, or worker protections'
        },
        'competition_oversupply': {
            'keywords': ['competition', 'driver', 'many', 'supply', 'demand', 'booking',
                        'order', 'trip', 'ride', 'wait', 'idle', 'queue', 'peak'],
            'description': 'Too many drivers, low demand, and intense competition'
        }
    }
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def extract_pain_points(self, posts: list[RedditPost]) -> list[PainPoint]:
        """Extract and rank pain points from posts."""
        
        category_data = defaultdict(lambda: {
            'mentions': 0,
            'sentiment_scores': [],
            'quotes': [],
            'weighted_score': 0
        })
        
        for post in posts:
            text = post.text.lower()
            sentiment = self.sentiment_analyzer.analyze_sentiment(post.text)
            
            # Only consider negative or neutral-negative content for pain points
            if sentiment['compound'] > 0.2:
                continue
            
            for category, data in self.PAIN_POINT_CATEGORIES.items():
                keywords_found = sum(1 for kw in data['keywords'] if kw in text)
                
                if keywords_found >= 2:  # At least 2 keywords must match
                    category_data[category]['mentions'] += 1
                    category_data[category]['sentiment_scores'].append(sentiment['compound'])
                    
                    # Weight by post score (popularity) and keyword density
                    weight = (1 + max(0, post.score) * 0.1) * keywords_found
                    category_data[category]['weighted_score'] += weight
                    
                    # Extract relevant quotes
                    negative_sentences = self.sentiment_analyzer.extract_negative_sentences(post.text)
                    for sentence in negative_sentences[:2]:
                        if len(sentence) > 30 and len(sentence) < 500:
                            category_data[category]['quotes'].append(sentence)
        
        # Create PainPoint objects
        pain_points = []
        for category, data in category_data.items():
            if data['mentions'] >= 3:  # Minimum threshold
                avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
                
                # Determine severity based on frequency and sentiment
                if data['mentions'] >= 15 and avg_sentiment < -0.3:
                    severity = 'high'
                elif data['mentions'] >= 8 or avg_sentiment < -0.4:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Select diverse example quotes
                unique_quotes = list(set(data['quotes']))[:5]
                
                pain_points.append(PainPoint(
                    category=category,
                    description=self.PAIN_POINT_CATEGORIES[category]['description'],
                    sentiment_score=avg_sentiment,
                    frequency=data['mentions'],
                    example_quotes=unique_quotes,
                    severity=severity
                ))
        
        # Sort by weighted importance (combination of frequency and sentiment)
        pain_points.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.severity],
            x.frequency,
            -x.sentiment_score  # More negative = higher priority
        ), reverse=True)
        
        return pain_points


def format_pain_point_report(pain_points: list[PainPoint], top_n: int = 5) -> str:
    """Format pain points into a readable report."""
    
    report = []
    report.append("=" * 70)
    report.append("DRIVER PAIN POINTS IN INDIA - ANALYSIS REPORT")
    report.append("Based on Reddit Discussions")
    report.append("=" * 70)
    report.append("")
    
    for i, pp in enumerate(pain_points[:top_n], 1):
        severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[pp.severity]
        
        report.append(f"{i}. {pp.category.upper().replace('_', ' ')}")
        report.append("-" * 50)
        report.append(f"   Severity: {severity_emoji} {pp.severity.upper()}")
        report.append(f"   Mentions: {pp.frequency} posts/comments")
        report.append(f"   Avg Sentiment: {pp.sentiment_score:.2f} (scale: -1 to +1)")
        report.append(f"   Description: {pp.description}")
        report.append("")
        
        if pp.example_quotes:
            report.append("   Sample Quotes:")
            for j, quote in enumerate(pp.example_quotes[:3], 1):
                # Clean and truncate quote
                clean_quote = quote.strip().replace('\n', ' ')
                if len(clean_quote) > 200:
                    clean_quote = clean_quote[:200] + "..."
                report.append(f"   {j}. \"{clean_quote}\"")
            report.append("")
        
        report.append("")
    
    report.append("=" * 70)
    report.append("METHODOLOGY")
    report.append("-" * 50)
    report.append("- Data Source: Reddit (Indian subreddits)")
    report.append("- Analysis: VADER Sentiment Analysis + Keyword Extraction")
    report.append("- Categories: Pre-defined based on common driver issues")
    report.append("- Severity: Based on mention frequency and sentiment intensity")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("=" * 50)
    print("Driver Pain Points Analyzer for India")
    print("=" * 50)
    print()
    
    # Step 1: Scrape Reddit
    print("[1/3] Connecting to Reddit API...")
    try:
        scraper = RedditScraper()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo set up Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Fill in: name, select 'script', redirect uri: http://localhost:8080")
        print("4. Copy the client ID (under app name) and secret")
        print("5. Create a .env file with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return
    
    print("[2/3] Searching for driver-related content...")
    posts = scraper.search_driver_content(limit=50)
    
    if not posts:
        print("\nNo driver-related content found. Try adjusting search parameters.")
        return
    
    # Step 2: Extract pain points
    print("[3/3] Analyzing sentiment and extracting pain points...")
    extractor = PainPointExtractor()
    pain_points = extractor.extract_pain_points(posts)
    
    if not pain_points:
        print("\nNo significant pain points identified from the data.")
        return
    
    # Step 3: Generate report
    print()
    report = format_pain_point_report(pain_points, top_n=5)
    print(report)
    
    # Save report to file
    report_file = "pain_points_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
