import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time

class CryptoDataCollector:
    def __init__(self):
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
        
    def get_price_data(self, days=30):
        """Collect crypto price data"""
        all_data = {}
        
        for symbol in self.crypto_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{days}d")
                all_data[symbol] = data
                print(f"✓ Collected {symbol} data: {len(data)} records")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"✗ Error collecting {symbol}: {e}")
                
        return all_data
    
    def get_news_data(self):
        """Simulate news data (replace with real API)"""
        # For demo - in real project, use NewsAPI, Reddit API, etc.
        sample_news = [
            {"date": "2024-01-15", "title": "Bitcoin reaches new highs", "sentiment": "positive"},
            {"date": "2024-01-14", "title": "Crypto market shows volatility", "sentiment": "neutral"},
            {"date": "2024-01-13", "title": "Regulatory concerns affect prices", "sentiment": "negative"},
            {"date": "2024-01-12", "title": "Institutional adoption grows", "sentiment": "positive"},
            {"date": "2024-01-11", "title": "Market correction continues", "sentiment": "negative"}
        ]
        
        return pd.DataFrame(sample_news)
    
    def save_data(self, price_data, news_data):
        """Save collected data"""
        # Save price data
        for symbol, data in price_data.items():
            data.to_csv(f'data/{symbol}_prices.csv')
            
        # Save news data
        news_data.to_csv('data/news_data.csv', index=False)
        print("✓ Data saved successfully!")

# Usage
if __name__ == "__main__":
    collector = CryptoDataCollector()
    
    print("🚀 Starting data collection...")
    price_data = collector.get_price_data(30)
    news_data = collector.get_news_data()
    
    collector.save_data(price_data, news_data)
    print("✅ Data collection complete!")