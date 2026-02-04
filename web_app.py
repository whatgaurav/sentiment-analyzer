#!/usr/bin/env python3
"""
Sentiment Analysis Web Application
A modern web UI for analyzing sentiment of text and Reddit posts.
"""

import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import nltk
import praw

# Setup NLTK
def setup_nltk():
    resources = ['vader_lexicon', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

setup_nltk()
load_dotenv()

# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))
CORS(app)  # Enable cross-origin requests from any website


class SentimentAnalyzer:
    """Performs sentiment analysis."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> dict:
        """Perform comprehensive sentiment analysis."""
        
        # VADER sentiment
        vader = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        # Sentence-level analysis
        sentences = sent_tokenize(text)
        sentence_sentiments = []
        for sent in sentences:
            sent_vader = self.sia.polarity_scores(sent)
            score = sent_vader['compound']
            if score >= 0.05:
                label = 'positive'
            elif score <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            sentence_sentiments.append({
                'text': sent,
                'score': round(score, 3),
                'label': label
            })
        
        # Overall label
        if vader['compound'] >= 0.05:
            overall_label = 'positive'
        elif vader['compound'] <= -0.05:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        # Subjectivity label
        if blob.sentiment.subjectivity < 0.4:
            subj_label = 'Objective'
        elif blob.sentiment.subjectivity > 0.6:
            subj_label = 'Subjective'
        else:
            subj_label = 'Mixed'
        
        return {
            'compound': round(vader['compound'], 3),
            'positive': round(vader['pos'] * 100, 1),
            'negative': round(vader['neg'] * 100, 1),
            'neutral': round(vader['neu'] * 100, 1),
            'polarity': round(blob.sentiment.polarity, 3),
            'subjectivity': round(blob.sentiment.subjectivity, 3),
            'subjectivity_label': subj_label,
            'overall_label': overall_label,
            'sentences': sentence_sentiments,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }


def fetch_reddit_post(url: str) -> tuple:
    """Fetch Reddit post content."""
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        return None, "Reddit API credentials not configured"
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='SentimentWebApp/1.0'
        )
        
        submission = reddit.submission(url=url)
        text = f"{submission.title}\n\n{submission.selftext}" if submission.selftext else submission.title
        
        metadata = {
            'title': submission.title,
            'subreddit': str(submission.subreddit),
            'score': submission.score,
            'num_comments': submission.num_comments
        }
        
        return {'text': text, 'metadata': metadata}, None
    except Exception as e:
        return None, str(e)


analyzer = SentimentAnalyzer()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of provided text."""
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if len(text) > 10000:
        return jsonify({'error': 'Text too long (max 10000 characters)'}), 400
    
    try:
        result = analyzer.analyze(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-reddit', methods=['POST'])
def analyze_reddit():
    """Analyze sentiment of a Reddit post."""
    data = request.json
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    if 'reddit.com' not in url and 'redd.it' not in url:
        return jsonify({'error': 'Invalid Reddit URL'}), 400
    
    post_data, error = fetch_reddit_post(url)
    
    if error:
        return jsonify({'error': error}), 400
    
    try:
        result = analyzer.analyze(post_data['text'])
        result['reddit_metadata'] = post_data['metadata']
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_ENV') != 'production'
    
    print("\n" + "=" * 50)
    print("  🔍 Sentiment Analysis Web App")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 50 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
