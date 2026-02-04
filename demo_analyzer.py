"""
Demo Driver Pain Points Analyzer
Demonstrates the sentiment analysis and pain point extraction using sample data.
No Reddit API credentials required.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

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


# Sample data representing typical Reddit posts about drivers in India
SAMPLE_POSTS = [
    # Low Earnings
    {
        "text": "I've been driving for Ola for 2 years now. The commission they take is insane - 25% plus GST. After fuel costs and car EMI, I barely make 15000 a month. It's impossible to survive in Mumbai with this income. The incentives have reduced drastically since 2019.",
        "score": 156,
        "subreddit": "mumbai"
    },
    {
        "text": "Uber driver here from Bangalore. The fare rates haven't increased since years but petrol price has doubled. We're making less money than before while working longer hours. Commission is killing us slowly.",
        "score": 89,
        "subreddit": "bangalore"
    },
    {
        "text": "My friend is an auto driver in Delhi. He says the CNG prices have gone up so much but the meter rates are still old. He works 12 hours and takes home barely 500 rupees profit. How is anyone supposed to live like this?",
        "score": 234,
        "subreddit": "delhi"
    },
    {
        "text": "Swiggy and Zomato have slashed delivery payments again. Earlier we used to get Rs 40 per delivery, now it's Rs 20-25. With rising fuel costs, most delivery boys are actually losing money on long distance orders.",
        "score": 178,
        "subreddit": "india"
    },
    
    # App/Platform Issues
    {
        "text": "The Ola app is absolute garbage. It crashes constantly, GPS doesn't work properly, and customer support is non-existent. I've lost so many rides because the app froze. When I complain, they just send automated responses.",
        "score": 67,
        "subreddit": "india"
    },
    {
        "text": "Uber's algorithm is completely unfair. It assigns rides to drivers who are further away while I'm sitting idle right next to the pickup point. No logic at all. The surge pricing never benefits drivers anymore.",
        "score": 145,
        "subreddit": "bangalore"
    },
    {
        "text": "Got deactivated from Rapido for no reason. Been driving for 3 years with 4.9 rating. They just sent a message saying account suspended. No explanation, no appeal process. Customer care doesn't pick up calls.",
        "score": 312,
        "subreddit": "chennai"
    },
    {
        "text": "Zomato's support is terrible. I had an issue with payment not credited for 2 weeks. Raised 15 tickets, all closed without resolution. They treat delivery partners like we're disposable. The app also shows wrong restaurant locations constantly.",
        "score": 98,
        "subreddit": "pune"
    },
    
    # Safety Concerns
    {
        "text": "My uncle is a cab driver. He was attacked by drunk passengers last week in South Delhi late night. They refused to pay and hit him when he asked for the fare. Police didn't help at all. It's so dangerous driving at night.",
        "score": 456,
        "subreddit": "delhi"
    },
    {
        "text": "Being a delivery driver in certain areas of Mumbai is genuinely scary. I've been robbed twice in the last 6 months. Swiggy and Zomato don't care about our safety at all. No insurance, no support.",
        "score": 234,
        "subreddit": "mumbai"
    },
    {
        "text": "Female Uber driver here. The harassment I face from male passengers is unbelievable. Inappropriate comments, requests for phone number, some even ask to meet after the trip. Reported to Uber multiple times, nothing happens.",
        "score": 567,
        "subreddit": "india"
    },
    {
        "text": "Accident happened while delivering food for Dunzo. Hospital bill was 45000 rupees. The company paid nothing. I had to sell my bike to cover medical expenses. No accident insurance for gig workers.",
        "score": 289,
        "subreddit": "bangalore"
    },
    
    # Work-Life Balance
    {
        "text": "To make decent money as an Ola driver, you need to work 14-16 hours daily. I haven't had a proper day off in 3 months. My health is suffering, hardly see my family. This is not sustainable.",
        "score": 178,
        "subreddit": "hyderabad"
    },
    {
        "text": "The targets set by Uber to get incentives are impossible. You need to complete 15 rides in peak hours to get Rs 500 extra. That means no breaks, no food, just drive drive drive. Mental stress is killing me.",
        "score": 134,
        "subreddit": "india"
    },
    {
        "text": "My father is a truck driver. He's on the road 20 days a month, barely sees us. The pressure from transport companies to meet deadlines means he drives without proper sleep. It's dangerous and unfair.",
        "score": 345,
        "subreddit": "india"
    },
    {
        "text": "Delivery partners have no fixed working hours. Swiggy expects you to be available during peak hours but doesn't guarantee minimum orders. I've waited for 4 hours without a single delivery sometimes.",
        "score": 89,
        "subreddit": "pune"
    },
    
    # Customer Behavior
    {
        "text": "Passengers these days are so rude. They cancel after I've driven 2km to pick them up. Or they rate 1 star because of traffic which is not my fault. My rating dropped from 4.8 to 4.3 because of unfair reviews.",
        "score": 234,
        "subreddit": "bangalore"
    },
    {
        "text": "Customer ordered food worth Rs 800, gave wrong address, then cancelled. I was stuck with the food and Zomato didn't reimburse me. How is this fair? We absorb all the losses while companies make profit.",
        "score": 456,
        "subreddit": "mumbai"
    },
    {
        "text": "The entitlement of some passengers is unbelievable. They expect AC at full blast, aux cable, water bottle, phone charger - basically a luxury service for Rs 100 ride. And if you say no, they threaten bad rating.",
        "score": 189,
        "subreddit": "india"
    },
    {
        "text": "Cash payment frauds are common. Customers say they'll pay cash, then claim they already paid online and run away. Ola customer care doesn't believe drivers, always side with the passenger.",
        "score": 267,
        "subreddit": "delhi"
    },
    
    # Vehicle Maintenance
    {
        "text": "Maintaining a car for Uber is so expensive. Service every 2 months, tyres every 6 months, insurance, permit renewal, fitness certificate. After all expenses, there's nothing left from earnings.",
        "score": 178,
        "subreddit": "india"
    },
    {
        "text": "Traffic police in Bangalore targets cab drivers. Got 3 challans this month for minor things. Rs 5000 gone just in fines. They know we can't afford to fight, so they exploit us.",
        "score": 234,
        "subreddit": "bangalore"
    },
    {
        "text": "My bike needs repair but I can't afford to take a day off because I have EMI to pay. So I ride with brakes that don't work properly. It's a cycle of debt and risk for delivery drivers.",
        "score": 145,
        "subreddit": "hyderabad"
    },
    
    # Lack of Benefits  
    {
        "text": "Gig workers have no rights in India. No minimum wage, no PF, no health insurance, no sick leave. If I get sick, I earn zero that day and have to pay for treatment myself. Companies treat us like machines.",
        "score": 567,
        "subreddit": "india"
    },
    {
        "text": "My colleague was hospitalized for a week. He's a full-time Swiggy delivery partner for 4 years. The company didn't pay a single rupee for his treatment or for the days he couldn't work.",
        "score": 345,
        "subreddit": "mumbai"
    },
    {
        "text": "There's no job security as a driver. One bad rating, one customer complaint, and you're deactivated. 5 years of service means nothing. No severance, no explanation, just gone.",
        "score": 234,
        "subreddit": "india"
    },
    
    # Competition/Oversupply
    {
        "text": "There are way too many drivers on the road now. In my area alone, there are 50 Ola/Uber drivers. Earlier I used to get 15 rides a day, now I'm lucky to get 6-7. Waiting time between rides is horrible.",
        "score": 156,
        "subreddit": "chennai"
    },
    {
        "text": "Zomato keeps adding more delivery partners even though existing ones don't get enough orders. It's a deliberate strategy to keep wages low. More competition means they can pay us less.",
        "score": 289,
        "subreddit": "bangalore"
    },
    {
        "text": "Peak hours are the only time to make money but everyone logs in during peak hours. So even peak is not guaranteed. Off-peak you might wait 2-3 hours for a single booking.",
        "score": 134,
        "subreddit": "pune"
    },
    
    # Some positive/neutral posts for balance
    {
        "text": "Started driving Uber 6 months ago. The flexibility is good, I can work when I want. But earnings are definitely lower than I expected. Need to figure out the optimal hours.",
        "score": 45,
        "subreddit": "india"
    },
    {
        "text": "Met a really nice Ola driver yesterday. He's been doing this for 8 years and shared some tips on how to maximize earnings. Sad to hear about the challenges they face though.",
        "score": 78,
        "subreddit": "bangalore"
    }
]


@dataclass
class RedditPost:
    """Represents a Reddit post or comment."""
    text: str
    score: int
    subreddit: str


@dataclass
class PainPoint:
    """Represents an identified pain point."""
    category: str
    description: str
    sentiment_score: float
    frequency: int
    example_quotes: list[str]
    severity: str


class SentimentAnalyzer:
    """Performs sentiment analysis on text."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment using VADER and TextBlob."""
        vader_scores = self.sia.polarity_scores(text)
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
    """Extracts and categorizes pain points."""
    
    PAIN_POINT_CATEGORIES = {
        'low_earnings': {
            'keywords': ['salary', 'earning', 'income', 'pay', 'money', 'commission', 
                        'incentive', 'fare', 'rate', 'minimum', 'wage', 'profit', 'loss',
                        'expensive', 'cost', 'fuel', 'petrol', 'diesel', 'cng', 'emi'],
            'description': 'Low earnings, high costs, and unfair commission structures'
        },
        'app_platform_issues': {
            'keywords': ['app', 'crash', 'bug', 'algorithm', 'allocation', 'assign',
                        'cancel', 'penalty', 'block', 'deactivate', 'suspend', 'ban',
                        'support', 'customer care', 'helpline', 'response', 'ticket'],
            'description': 'Issues with ride-hailing apps, algorithms, and platform support'
        },
        'safety_concerns': {
            'keywords': ['safety', 'safe', 'danger', 'attack', 'assault', 'robbery',
                        'threat', 'harassment', 'abuse', 'violence', 'accident', 'risk',
                        'night', 'robbed', 'police', 'scared', 'scary'],
            'description': 'Safety risks, harassment, and dangerous situations'
        },
        'work_life_balance': {
            'keywords': ['hour', 'time', 'long', 'shift', 'rest', 'break', 'tired',
                        'exhausted', 'family', 'health', 'stress', 'pressure', 'sleep',
                        'overtime', 'target', 'quota', 'sustainable'],
            'description': 'Long working hours, stress, and lack of work-life balance'
        },
        'customer_behavior': {
            'keywords': ['passenger', 'customer', 'rider', 'rude', 'cancel', 'rating',
                        'review', 'complaint', 'behavior', 'respect', 'argue', 'fight',
                        'tip', 'cash', 'payment', 'fraud', 'entitlement', 'star'],
            'description': 'Difficult customers, unfair ratings, and payment issues'
        },
        'vehicle_maintenance': {
            'keywords': ['vehicle', 'car', 'bike', 'maintenance', 'repair', 'service',
                        'emi', 'loan', 'insurance', 'permit', 'document', 'fine',
                        'traffic', 'challan', 'police', 'tyre', 'fitness'],
            'description': 'Vehicle costs, maintenance burden, and regulatory issues'
        },
        'lack_of_benefits': {
            'keywords': ['insurance', 'health', 'medical', 'benefit', 'sick', 'leave',
                        'holiday', 'protection', 'rights', 'union', 'contract', 'job',
                        'security', 'permanent', 'gig', 'pf', 'hospitalized'],
            'description': 'No job security, health benefits, or worker protections'
        },
        'competition_oversupply': {
            'keywords': ['competition', 'many', 'supply', 'demand', 'booking',
                        'order', 'trip', 'ride', 'wait', 'idle', 'queue', 'peak', 'waiting'],
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
            
            # Only consider negative or neutral-negative content
            if sentiment['compound'] > 0.2:
                continue
            
            for category, data in self.PAIN_POINT_CATEGORIES.items():
                keywords_found = sum(1 for kw in data['keywords'] if kw in text)
                
                if keywords_found >= 2:
                    category_data[category]['mentions'] += 1
                    category_data[category]['sentiment_scores'].append(sentiment['compound'])
                    
                    weight = (1 + max(0, post.score) * 0.1) * keywords_found
                    category_data[category]['weighted_score'] += weight
                    
                    negative_sentences = self.sentiment_analyzer.extract_negative_sentences(post.text)
                    for sentence in negative_sentences[:2]:
                        if len(sentence) > 30 and len(sentence) < 500:
                            category_data[category]['quotes'].append(sentence)
        
        pain_points = []
        for category, data in category_data.items():
            if data['mentions'] >= 2:
                avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
                
                if data['mentions'] >= 8 and avg_sentiment < -0.3:
                    severity = 'high'
                elif data['mentions'] >= 4 or avg_sentiment < -0.4:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                unique_quotes = list(set(data['quotes']))[:5]
                
                pain_points.append(PainPoint(
                    category=category,
                    description=self.PAIN_POINT_CATEGORIES[category]['description'],
                    sentiment_score=avg_sentiment,
                    frequency=data['mentions'],
                    example_quotes=unique_quotes,
                    severity=severity
                ))
        
        pain_points.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.severity],
            x.frequency,
            -x.sentiment_score
        ), reverse=True)
        
        return pain_points


def format_pain_point_report(pain_points: list[PainPoint], top_n: int = 5) -> str:
    """Format pain points into a readable report."""
    
    report = []
    report.append("=" * 70)
    report.append("DRIVER PAIN POINTS IN INDIA - ANALYSIS REPORT")
    report.append("Based on Reddit Discussions (Demo Data)")
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
                clean_quote = quote.strip().replace('\n', ' ')
                if len(clean_quote) > 200:
                    clean_quote = clean_quote[:200] + "..."
                report.append(f"   {j}. \"{clean_quote}\"")
            report.append("")
        
        report.append("")
    
    report.append("=" * 70)
    report.append("METHODOLOGY")
    report.append("-" * 50)
    report.append("- Data Source: Sample Reddit posts (demo mode)")
    report.append("- Analysis: VADER Sentiment Analysis + Keyword Extraction")
    report.append("- Categories: Pre-defined based on common driver issues")
    report.append("- Severity: Based on mention frequency and sentiment intensity")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("=" * 50)
    print("Driver Pain Points Analyzer - DEMO MODE")
    print("Using sample data (no Reddit API needed)")
    print("=" * 50)
    print()
    
    # Convert sample data to RedditPost objects
    print("[1/2] Loading sample data...")
    posts = [
        RedditPost(
            text=post["text"],
            score=post["score"],
            subreddit=post["subreddit"]
        )
        for post in SAMPLE_POSTS
    ]
    print(f"  Loaded {len(posts)} sample posts")
    
    # Extract pain points
    print("[2/2] Analyzing sentiment and extracting pain points...")
    extractor = PainPointExtractor()
    pain_points = extractor.extract_pain_points(posts)
    
    # Generate report
    print()
    report = format_pain_point_report(pain_points, top_n=5)
    print(report)
    
    # Save report
    report_file = "pain_points_demo_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    print("\n" + "=" * 50)
    print("To analyze live Reddit data, run 'python analyzer.py'")
    print("after setting up your Reddit API credentials.")
    print("=" * 50)


if __name__ == "__main__":
    main()
