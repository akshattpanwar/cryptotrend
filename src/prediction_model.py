import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class CryptoPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_features(self, price_df, sentiment_df=None):
        """Create features for prediction"""
        features = []
        
        # Technical indicators
        price_df['SMA_7'] = price_df['Close'].rolling(window=7).mean()
        price_df['SMA_14'] = price_df['Close'].rolling(window=14).mean()
        price_df['RSI'] = self.calculate_rsi(price_df['Close'])
        price_df['Volatility'] = price_df['Close'].rolling(window=7).std()
        
        # Price features
        price_df['Price_Change'] = price_df['Close'].pct_change()
        price_df['Volume_Change'] = price_df['Volume'].pct_change()
        
        feature_columns = ['SMA_7', 'SMA_14', 'RSI', 'Volatility', 
                          'Price_Change', 'Volume_Change', 'Volume']
        
        # Add sentiment if available
        if sentiment_df is not None:
            price_df = self.merge_sentiment_data(price_df, sentiment_df)
            feature_columns.extend(['avg_sentiment', 'news_count'])
            
        return price_df[feature_columns].fillna(0)
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def merge_sentiment_data(self, price_df, sentiment_df):
        """Merge sentiment data with price data"""
        price_df.reset_index(inplace=True)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        merged = price_df.merge(sentiment_df, left_on='Date', right_on='date', how='left')
        merged[['avg_sentiment', 'news_count']] = merged[['avg_sentiment', 'news_count']].fillna(0)
        
        return merged.set_index('Date')
    
    def prepare_data(self, features, target, test_size=0.2):
        """Prepare data for training"""
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X_clean = features[valid_idx]
        y_clean = target[valid_idx]
        
        # Split data
        split_idx = int(len(X_clean) * (1 - test_size))
        
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the prediction model"""
        self.model.fit(X_train, y_train)
        print("✓ Model trained successfully!")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2_Score': r2
        }
        
        return metrics, predictions
    
    def save_model(self):
        """Save trained model"""
        joblib.dump(self.model, 'results/crypto_model.pkl')
        joblib.dump(self.scaler, 'results/scaler.pkl')
        print("✓ Model saved!")

# Usage
if __name__ == "__main__":
    predictor = CryptoPredictionModel()
    
    print("🤖 Starting model training...")
    
    # Load data
    btc_data = pd.read_csv('data/BTC-USD_prices.csv', index_col=0, parse_dates=True)
    sentiment_data = pd.read_csv('results/daily_sentiment.csv')
    
    # Create features
    features = predictor.create_features(btc_data, sentiment_data)
    target = btc_data['Close'].shift(-1)  # Predict next day price
    
    # Prepare and train
    X_train, X_test, y_train, y_test = predictor.prepare_data(features, target)
    predictor.train_model(X_train, y_train)
    
    # Evaluate
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    
    print(f"📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    predictor.save_model()
    print("✅ Model training complete!")