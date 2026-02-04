#!/usr/bin/env python3
"""
Interactive Sentiment Analysis Tool
Analyze sentiment of any Reddit post URL or custom text.
"""

import os
import re
import sys
from dataclasses import dataclass

import praw
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import nltk

# Setup NLTK
def setup_nltk():
    resources = ['vader_lexicon', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

setup_nltk()


@dataclass
class SentimentResult:
    """Stores sentiment analysis results."""
    text: str
    compound_score: float
    positive: float
    negative: float
    neutral: float
    polarity: float
    subjectivity: float
    sentence_sentiments: list


class SentimentAnalyzer:
    """Performs detailed sentiment analysis."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> SentimentResult:
        """Perform comprehensive sentiment analysis on text."""
        
        # VADER sentiment
        vader = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        # Sentence-level analysis
        sentences = sent_tokenize(text)
        sentence_sentiments = []
        for sent in sentences:
            sent_vader = self.sia.polarity_scores(sent)
            sentence_sentiments.append({
                'text': sent,
                'score': sent_vader['compound'],
                'label': self._get_label(sent_vader['compound'])
            })
        
        return SentimentResult(
            text=text,
            compound_score=vader['compound'],
            positive=vader['pos'],
            negative=vader['neg'],
            neutral=vader['neu'],
            polarity=blob.sentiment.polarity,
            subjectivity=blob.sentiment.subjectivity,
            sentence_sentiments=sentence_sentiments
        )
    
    def _get_label(self, score: float) -> str:
        """Convert score to sentiment label."""
        if score >= 0.05:
            return '✅ Positive'
        elif score <= -0.05:
            return '❌ Negative'
        else:
            return '➖ Neutral'


def get_sentiment_bar(score: float, width: int = 40) -> str:
    """Create a visual bar for sentiment score."""
    # Score ranges from -1 to 1, normalize to 0-width
    normalized = int((score + 1) / 2 * width)
    bar = '█' * normalized + '░' * (width - normalized)
    return f"[{bar}]"


def format_results(result: SentimentResult) -> str:
    """Format sentiment results for display."""
    
    output = []
    output.append("\n" + "=" * 60)
    output.append("SENTIMENT ANALYSIS RESULTS")
    output.append("=" * 60)
    
    # Overall sentiment
    label = "✅ POSITIVE" if result.compound_score >= 0.05 else "❌ NEGATIVE" if result.compound_score <= -0.05 else "➖ NEUTRAL"
    output.append(f"\n📊 OVERALL SENTIMENT: {label}")
    output.append(f"   Compound Score: {result.compound_score:.3f}")
    output.append(f"   {get_sentiment_bar(result.compound_score)}")
    output.append(f"   -1 (Negative) ←────────────→ +1 (Positive)")
    
    # Detailed scores
    output.append("\n📈 DETAILED BREAKDOWN:")
    output.append(f"   Positive:    {result.positive:.1%} {'█' * int(result.positive * 20)}")
    output.append(f"   Negative:    {result.negative:.1%} {'█' * int(result.negative * 20)}")
    output.append(f"   Neutral:     {result.neutral:.1%} {'█' * int(result.neutral * 20)}")
    
    # TextBlob analysis
    output.append("\n🔍 TEXTBLOB ANALYSIS:")
    output.append(f"   Polarity:    {result.polarity:.3f} (negative ← → positive)")
    subj_label = "Objective" if result.subjectivity < 0.4 else "Subjective" if result.subjectivity > 0.6 else "Mixed"
    output.append(f"   Subjectivity: {result.subjectivity:.3f} ({subj_label})")
    
    # Sentence breakdown
    if len(result.sentence_sentiments) > 1:
        output.append("\n📝 SENTENCE-BY-SENTENCE ANALYSIS:")
        output.append("-" * 60)
        for i, sent in enumerate(result.sentence_sentiments, 1):
            text = sent['text'][:80] + "..." if len(sent['text']) > 80 else sent['text']
            output.append(f"\n   {i}. {sent['label']} (score: {sent['score']:.2f})")
            output.append(f"      \"{text}\"")
    
    # Interpretation
    output.append("\n" + "=" * 60)
    output.append("📋 INTERPRETATION:")
    output.append("-" * 60)
    
    if result.compound_score >= 0.5:
        output.append("   The text expresses strongly positive sentiment.")
    elif result.compound_score >= 0.05:
        output.append("   The text expresses mildly positive sentiment.")
    elif result.compound_score <= -0.5:
        output.append("   The text expresses strongly negative sentiment.")
    elif result.compound_score <= -0.05:
        output.append("   The text expresses mildly negative sentiment.")
    else:
        output.append("   The text is largely neutral in sentiment.")
    
    if result.subjectivity > 0.6:
        output.append("   The content is highly subjective/opinion-based.")
    elif result.subjectivity < 0.4:
        output.append("   The content is relatively objective/factual.")
    
    # Count positive/negative sentences
    pos_count = sum(1 for s in result.sentence_sentiments if s['score'] >= 0.05)
    neg_count = sum(1 for s in result.sentence_sentiments if s['score'] <= -0.05)
    if len(result.sentence_sentiments) > 1:
        output.append(f"   {pos_count} positive, {neg_count} negative sentences out of {len(result.sentence_sentiments)}.")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def fetch_reddit_post(url: str) -> tuple[str, dict]:
    """Fetch Reddit post content from URL."""
    
    load_dotenv()
    
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError(
            "Reddit API credentials not found.\n"
            "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file.\n"
            "Get credentials at: https://www.reddit.com/prefs/apps"
        )
    
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent='SentimentTool/1.0'
    )
    
    # Extract submission from URL
    submission = reddit.submission(url=url)
    
    # Combine title and body
    text = f"{submission.title}\n\n{submission.selftext}" if submission.selftext else submission.title
    
    metadata = {
        'title': submission.title,
        'subreddit': str(submission.subreddit),
        'score': submission.score,
        'num_comments': submission.num_comments,
        'url': url
    }
    
    return text, metadata


def print_header():
    """Print tool header."""
    print("\n" + "=" * 60)
    print("  🔍 SENTIMENT ANALYSIS TOOL")
    print("  Analyze any Reddit post or custom text")
    print("=" * 60)


def print_menu():
    """Print menu options."""
    print("\nOptions:")
    print("  1. Analyze a Reddit post URL")
    print("  2. Analyze custom text")
    print("  3. Analyze from file")
    print("  q. Quit")
    print()


def analyze_reddit_url():
    """Handle Reddit URL analysis."""
    url = input("\n📎 Enter Reddit post URL: ").strip()
    
    if not url:
        print("❌ No URL provided.")
        return
    
    # Validate URL format
    if 'reddit.com' not in url and 'redd.it' not in url:
        print("❌ Invalid Reddit URL. Please provide a valid Reddit post URL.")
        return
    
    print("\n⏳ Fetching Reddit post...")
    
    try:
        text, metadata = fetch_reddit_post(url)
        
        print("\n📌 POST DETAILS:")
        print(f"   Subreddit: r/{metadata['subreddit']}")
        print(f"   Title: {metadata['title'][:60]}...")
        print(f"   Score: {metadata['score']} | Comments: {metadata['num_comments']}")
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(text)
        print(format_results(result))
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Failed to fetch post: {e}")


def analyze_custom_text():
    """Handle custom text analysis."""
    print("\n📝 Enter your text (press Enter twice to finish):")
    print("-" * 40)
    
    lines = []
    empty_count = 0
    
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    
    text = "\n".join(lines).strip()
    
    if not text:
        print("❌ No text provided.")
        return
    
    print("\n⏳ Analyzing...")
    
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text)
    print(format_results(result))


def analyze_from_file():
    """Analyze text from a file."""
    filepath = input("\n📁 Enter file path: ").strip()
    
    if not filepath:
        print("❌ No file path provided.")
        return
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print("❌ File is empty.")
            return
        
        print(f"\n⏳ Analyzing {len(text)} characters...")
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(text)
        print(format_results(result))
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def quick_analyze(text: str):
    """Quick analysis for command line argument."""
    print_header()
    print("\n⏳ Analyzing provided text...")
    
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text)
    print(format_results(result))


def main():
    """Main interactive loop."""
    
    # Check for command line argument
    if len(sys.argv) > 1:
        # If argument is a URL, fetch and analyze
        arg = " ".join(sys.argv[1:])
        if 'reddit.com' in arg or 'redd.it' in arg:
            print_header()
            print(f"\n📎 Analyzing URL: {arg}")
            try:
                text, metadata = fetch_reddit_post(arg)
                print(f"\n📌 Post from r/{metadata['subreddit']}: {metadata['title'][:50]}...")
                analyzer = SentimentAnalyzer()
                result = analyzer.analyze(text)
                print(format_results(result))
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            # Treat as text to analyze
            quick_analyze(arg)
        return
    
    # Interactive mode
    print_header()
    
    while True:
        print_menu()
        choice = input("Select option: ").strip().lower()
        
        if choice == '1':
            analyze_reddit_url()
        elif choice == '2':
            analyze_custom_text()
        elif choice == '3':
            analyze_from_file()
        elif choice in ('q', 'quit', 'exit'):
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
