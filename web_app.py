#!/usr/bin/env python3
"""
Sentiment Analysis Web Application
A modern web UI for analyzing sentiment of text and Reddit posts.
Supports 100+ languages via hybrid VADER + HuggingFace approach.
"""

import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from langdetect import detect, LangDetectException
import nltk
import praw

# Setup NLTK (only need tokenizers now, VADER is standalone)
def setup_nltk():
    resources = ['punkt', 'punkt_tab']
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

# Language names for display
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
    'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
    'bn': 'Bengali', 'pa': 'Punjabi', 'te': 'Telugu', 'mr': 'Marathi',
    'ta': 'Tamil', 'ur': 'Urdu', 'gu': 'Gujarati', 'kn': 'Kannada',
    'ml': 'Malayalam', 'th': 'Thai', 'vi': 'Vietnamese', 'tr': 'Turkish',
    'pl': 'Polish', 'uk': 'Ukrainian', 'ro': 'Romanian', 'el': 'Greek',
    'hu': 'Hungarian', 'cs': 'Czech', 'sv': 'Swedish', 'da': 'Danish',
    'fi': 'Finnish', 'no': 'Norwegian', 'he': 'Hebrew', 'id': 'Indonesian',
    'ms': 'Malay', 'tl': 'Filipino', 'sw': 'Swahili', 'af': 'Afrikaans',
}


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer supporting 100+ languages.
    - English: Fast local VADER analysis
    - Other languages: HuggingFace Inference API
    - Fallback: VADER if API fails
    """
    
    # HuggingFace Inference API (free, no key required for public models)
    # Using nlptown's popular multilingual sentiment model (1-5 stars)
    HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def detect_language(self, text: str) -> tuple:
        """Detect language of text. Returns (code, name)."""
        try:
            code = detect(text)
            name = LANGUAGE_NAMES.get(code, code.upper())
            return code, name
        except LangDetectException:
            return 'en', 'English'
    
    def analyze_with_huggingface(self, text: str) -> tuple:
        """
        Analyze sentiment using HuggingFace multilingual model.
        Returns (result_dict, error_message) - result is None if API fails.
        """
        try:
            response = requests.post(
                self.HF_API_URL,
                json={"inputs": text[:512]},  # Model max length
                timeout=30
            )
            
            if response.status_code != 200:
                return None, f"API returned status {response.status_code}"
            
            results = response.json()
            
            # Handle model loading state
            if isinstance(results, dict) and 'error' in results:
                return None, f"API error: {results.get('error', 'unknown')}"
            
            # Handle estimated_time (model is loading)
            if isinstance(results, dict) and 'estimated_time' in results:
                return None, f"Model loading, wait {results.get('estimated_time', '?')}s"
            
            # Parse results - handle nested list format [[{label, score}, ...]]
            if results and isinstance(results, list):
                # Handle nested list [[...]]
                items = results[0] if isinstance(results[0], list) else results
                
                scores = {}
                for item in items:
                    if isinstance(item, dict):
                        label = item.get('label', '').lower()
                        score = item.get('score', 0)
                        scores[label] = score
                
                # Handle different model formats:
                # 1. Direct labels: positive, negative, neutral
                # 2. LABEL_X format: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
                # 3. Star ratings: 1 star, 2 stars, ..., 5 stars (nlptown model)
                
                if 'positive' in scores or 'negative' in scores:
                    pos = scores.get('positive', 0)
                    neg = scores.get('negative', 0)
                    neu = scores.get('neutral', 0)
                elif 'label_0' in scores:
                    pos = scores.get('label_2', 0)
                    neg = scores.get('label_0', 0)
                    neu = scores.get('label_1', 0)
                elif '5 stars' in scores or '1 star' in scores:
                    # nlptown model uses star ratings
                    # Convert to positive/negative/neutral
                    s1 = scores.get('1 star', 0)
                    s2 = scores.get('2 stars', 0)
                    s3 = scores.get('3 stars', 0)
                    s4 = scores.get('4 stars', 0)
                    s5 = scores.get('5 stars', 0)
                    
                    neg = s1 + s2  # 1-2 stars = negative
                    neu = s3       # 3 stars = neutral
                    pos = s4 + s5  # 4-5 stars = positive
                else:
                    return None, f"Unknown label format: {list(scores.keys())}"
                
                # Calculate compound score similar to VADER (-1 to 1)
                compound = pos - neg
                
                return {
                    'compound': round(compound, 3),
                    'positive': round(pos * 100, 1),
                    'negative': round(neg * 100, 1),
                    'neutral': round(neu * 100, 1),
                    'debug': f"scores={scores}",
                }, None
            
            return None, f"Unexpected format: {type(results).__name__}"
            
        except requests.exceptions.Timeout:
            return None, "API timeout (30s)"
        except requests.exceptions.ConnectionError:
            return None, "Connection error"
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def analyze_with_vader(self, text: str) -> dict:
        """Analyze sentiment using local VADER."""
        vader = self.sia.polarity_scores(text)
        return {
            'compound': round(vader['compound'], 3),
            'positive': round(vader['pos'] * 100, 1),
            'negative': round(vader['neg'] * 100, 1),
            'neutral': round(vader['neu'] * 100, 1),
        }
    
    def analyze(self, text: str) -> dict:
        """
        Perform comprehensive sentiment analysis.
        Uses VADER for English, HuggingFace for other languages.
        """
        # Detect language
        lang_code, lang_name = self.detect_language(text)
        is_english = lang_code == 'en'
        
        # Choose analysis method
        hf_error = None
        if is_english:
            sentiment = self.analyze_with_vader(text)
            analysis_method = 'VADER (local)'
        else:
            # Try HuggingFace for non-English
            sentiment, hf_error = self.analyze_with_huggingface(text)
            if sentiment:
                analysis_method = 'XLM-RoBERTa (multilingual)'
            else:
                # Fallback to VADER if API fails
                sentiment = self.analyze_with_vader(text)
                analysis_method = f'VADER (fallback: {hf_error})'
        
        # TextBlob for subjectivity (works best for English)
        blob = TextBlob(text)
        
        # Sentence-level analysis (VADER for all, fast enough)
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
        
        sentence_sentiments = []
        for sent in sentences:
            if not sent.strip():
                continue
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
        if sentiment['compound'] >= 0.05:
            overall_label = 'positive'
        elif sentiment['compound'] <= -0.05:
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
            'compound': sentiment['compound'],
            'positive': sentiment['positive'],
            'negative': sentiment['negative'],
            'neutral': sentiment['neutral'],
            'polarity': round(blob.sentiment.polarity, 3),
            'subjectivity': round(blob.sentiment.subjectivity, 3),
            'subjectivity_label': subj_label,
            'overall_label': overall_label,
            'sentences': sentence_sentiments,
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'language': lang_name,
            'language_code': lang_code,
            'analysis_method': analysis_method,
            'debug_info': sentiment.get('debug', None),
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


analyzer = HybridSentimentAnalyzer()


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
