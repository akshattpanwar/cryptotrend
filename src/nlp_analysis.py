import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import Counter
import re

class CryptoNLPAnalyzer:
    def __init__(self):
        self.positive_words = ['bull', 'moon', 'pump', 'rise', 'gain', 'profit', 'buy']
        self.negative_words = ['bear', 'dump', 'crash', 'fall', 'loss', 'sell', 'dip']
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_sentiment_score(self, text):
        """Calculate sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # Custom crypto sentiment boost
        crypto_boost = 0
        words = text.lower().split()
        
        for word in words:
            if word in self.positive_words:
                crypto_boost += 0.2
            elif word in self.negative_words:
                crypto_boost -= 0.2
                
        final_score = np.clip(polarity + crypto_boost, -1, 1)
        return final_score
    
    def categorize_sentiment(self, score):
        """Convert score to category"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment of news data"""
        results = []
        
        for _, row in news_df.iterrows():
            clean_title = self.clean_text(row['title'])
            sentiment_score = self.get_sentiment_score(clean_title)
            sentiment_category = self.categorize_sentiment(sentiment_score)
            
            results.append({
                'date': row['date'],
                'title': row['title'],
                'clean_title': clean_title,
                'sentiment_score': sentiment_score,
                'sentiment_category': sentiment_category
            })
            
        return pd.DataFrame(results)
    
    def get_daily_sentiment(self, sentiment_df):
        """Aggregate daily sentiment scores"""
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment_category': lambda x: Counter(x).most_common(1)[0][0]
        }).round(3)
        
        daily_sentiment.columns = ['avg_sentiment', 'news_count', 'dominant_sentiment']
        return daily_sentiment.reset_index()
    
    def save_results(self, sentiment_df, daily_sentiment):
        """Save analysis results"""
        sentiment_df.to_csv('results/sentiment_analysis.csv', index=False)
        daily_sentiment.to_csv('results/daily_sentiment.csv', index=False)
        print("✓ NLP analysis results saved!")

# Usage
if __name__ == "__main__":
    analyzer = CryptoNLPAnalyzer()
    
    print("🧠 Starting NLP analysis...")
    news_df = pd.read_csv('data/news_data.csv')
    
    sentiment_results = analyzer.analyze_news_sentiment(news_df)
    daily_sentiment = analyzer.get_daily_sentiment(sentiment_results)
    
    analyzer.save_results(sentiment_results, daily_sentiment)
    print("✅ NLP analysis complete!")