import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

class CryptoVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_price_trends(self, price_data):
        """Plot cryptocurrency price trends"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cryptocurrency Price Analysis', fontsize=16, fontweight='bold')
        
        # Price trend
        axes[0,0].plot(price_data.index, price_data['Close'], linewidth=2, color='#2E86AB')
        axes[0,0].set_title('BTC Price Trend', fontweight='bold')
        axes[0,0].set_ylabel('Price (USD)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Volume
        axes[0,1].bar(price_data.index, price_data['Volume'], alpha=0.7, color='#A23B72')
        axes[0,1].set_title('Trading Volume', fontweight='bold')
        axes[0,1].set_ylabel('Volume')
        axes[0,1].grid(True, alpha=0.3)
        
        # Daily returns
        returns = price_data['Close'].pct_change().dropna()
        axes[1,0].hist(returns, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
        axes[1,0].set_title('Daily Returns Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Daily Return (%)')
        axes[1,0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}')
        axes[1,0].legend()
        
        # Moving averages
        price_data['SMA_7'] = price_data['Close'].rolling(7).mean()
        price_data['SMA_14'] = price_data['Close'].rolling(14).mean()
        
        axes[1,1].plot(price_data.index, price_data['Close'], label='Price', linewidth=2)
        axes[1,1].plot(price_data.index, price_data['SMA_7'], label='SMA 7', linestyle='--')
        axes[1,1].plot(price_data.index, price_data['SMA_14'], label='SMA 14', linestyle='--')
        axes[1,1].set_title('Price with Moving Averages', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_sentiment_analysis(self, sentiment_data, daily_sentiment):
        """Plot sentiment analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # Sentiment distribution
        sentiment_counts = sentiment_data['sentiment_category'].value_counts()
        colors = ['#27AE60', '#E74C3C', '#F39C12']  # green, red, orange
        
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0,0].set_title('Sentiment Distribution', fontweight='bold')
        
        # Daily sentiment trend
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        axes[0,1].plot(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 
                       marker='o', linewidth=2, color='#8E44AD')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,1].set_title('Daily Sentiment Trend', fontweight='bold')
        axes[0,1].set_ylabel('Average Sentiment Score')
        axes[0,1].grid(True, alpha=0.3)
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
        
        # Sentiment score histogram
        axes[1,0].hist(sentiment_data['sentiment_score'], bins=20, alpha=0.7, 
                       color='#3498DB', edgecolor='black')
        axes[1,0].set_title('Sentiment Score Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Sentiment Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(sentiment_data['sentiment_score'].mean(), 
                          color='red', linestyle='--', label='Mean')
        axes[1,0].legend()
        
        # News count per day
        axes[1,1].bar(daily_sentiment['date'], daily_sentiment['news_count'], 
                      alpha=0.7, color='#E67E22')
        axes[1,1].set_title('News Count per Day', fontweight='bold')
        axes[1,1].set_ylabel('Number of News')
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_results(self, actual, predicted):
        """Plot prediction model results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Prediction Model Results', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted
        axes[0].scatter(actual, predicted, alpha=0.6, color='#3498DB')
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Prices')
        axes[0].set_ylabel('Predicted Prices')
        axes[0].set_title('Actual vs Predicted Prices', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = actual - predicted
        axes[1].scatter(predicted, residuals, alpha=0.6, color='#E74C3C')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('Predicted Prices')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_report(self, metrics):
        """Create a summary report"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a text summary
        summary_text = f"""
        CRYPTO TREND ANALYSIS - SUMMARY REPORT
        =====================================
        
        📊 Model Performance Metrics:
        • R² Score: {metrics['R2_Score']:.4f}
        • RMSE: ${metrics['RMSE']:.2f}
        • MSE: {metrics['MSE']:.2f}
        
        🔍 Key Insights:
        • Model explains {metrics['R2_Score']*100:.1f}% of price variance
        • Average prediction error: ${metrics['RMSE']:.2f}
        • Sentiment analysis incorporated successfully
        
        📈 Technical Features Used:
        • Moving Averages (SMA 7, 14)
        • RSI (Relative Strength Index)
        • Price volatility
        • Volume changes
        • Sentiment scores from NLP analysis
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        ax.axis('off')
        
        plt.savefig('results/summary_report.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage
if __name__ == "__main__":
    visualizer = CryptoVisualizer()
    
    print("📊 Creating visualizations...")
    
    # Load all data
    btc_data = pd.read_csv('data/BTC-USD_prices.csv', index_col=0, parse_dates=True)
    sentiment_data = pd.read_csv('results/sentiment_analysis.csv')
    daily_sentiment = pd.read_csv('results/daily_sentiment.csv')
    
    # Create all plots
    visualizer.plot_price_trends(btc_data)
    visualizer.plot_sentiment_analysis(sentiment_data, daily_sentiment)
    
    # Example metrics (replace with actual model results)
    example_metrics = {'R2_Score': 0.75, 'RMSE': 1250.50, 'MSE': 1563750.25}
    visualizer.create_summary_report(example_metrics)
    
    print("✅ Visualization complete!")