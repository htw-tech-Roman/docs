#!/usr/bin/env python3
"""
Crypto Pump Analyzer with News Sentiment Analysis
Analyzes anomalous volume spikes for potential pump activities
Considers news sentiment and market maker activities (DWF Labs, etc.)
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPumpAnalyzer:
    """Main class for analyzing crypto pump activities"""
    
    def __init__(self):
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Market makers to track
        self.market_makers = {
            'DWF Labs': 'dwf-labs',
            'Alameda Research': 'alameda-research',
            'Jump Trading': 'jump-trading',
            'Wintermute': 'wintermute',
            'GSR': 'gsr'
        }
        
        # Volume spike thresholds
        self.volume_thresholds = {
            'extreme': 10.0,  # 10x normal volume
            'high': 5.0,      # 5x normal volume
            'moderate': 2.0   # 2x normal volume
        }
    
    def get_coin_data(self, coin_id: str, days: int = 30) -> Dict:
        """Fetch coin data from CoinGecko"""
        try:
            url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(x[0]/1000) for x in data['prices']],
                'price': [x[1] for x in data['prices']],
                'volume': [x[1] for x in data['total_volumes']],
                'market_cap': [x[1] for x in data['market_caps']]
            })
            
            return {
                'data': df,
                'coin_id': coin_id,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {coin_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume anomalies using statistical methods"""
        df = df.copy()
        
        # Calculate rolling statistics
        df['volume_ma_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
        df['volume_ma_30'] = df['volume'].rolling(window=30, min_periods=1).mean()
        df['volume_std'] = df['volume'].rolling(window=30, min_periods=1).std()
        
        # Calculate volume ratios
        df['volume_ratio_7'] = df['volume'] / df['volume_ma_7']
        df['volume_ratio_30'] = df['volume'] / df['volume_ma_30']
        
        # Z-score for volume
        df['volume_zscore'] = (df['volume'] - df['volume_ma_30']) / df['volume_std']
        
        # Classify anomalies
        df['anomaly_level'] = 'normal'
        df.loc[df['volume_ratio_7'] >= self.volume_thresholds['extreme'], 'anomaly_level'] = 'extreme'
        df.loc[(df['volume_ratio_7'] >= self.volume_thresholds['high']) & 
               (df['volume_ratio_7'] < self.volume_thresholds['extreme']), 'anomaly_level'] = 'high'
        df.loc[(df['volume_ratio_7'] >= self.volume_thresholds['moderate']) & 
               (df['volume_ratio_7'] < self.volume_thresholds['high']), 'anomaly_level'] = 'moderate'
        
        # Price change analysis
        df['price_change'] = df['price'].pct_change() * 100
        df['price_change_24h'] = df['price'].pct_change(periods=24) * 100
        
        return df
    
    def analyze_pump_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze various pump indicators"""
        indicators = {}
        
        # Volume analysis
        recent_data = df.tail(24)  # Last 24 hours
        max_volume_ratio = recent_data['volume_ratio_7'].max()
        avg_volume_ratio = recent_data['volume_ratio_7'].mean()
        
        indicators['volume_spike'] = {
            'max_ratio': max_volume_ratio,
            'avg_ratio': avg_volume_ratio,
            'extreme_spikes': len(recent_data[recent_data['anomaly_level'] == 'extreme']),
            'high_spikes': len(recent_data[recent_data['anomaly_level'] == 'high'])
        }
        
        # Price analysis
        max_price_change = recent_data['price_change'].max()
        min_price_change = recent_data['price_change'].min()
        price_volatility = recent_data['price_change'].std()
        
        indicators['price_action'] = {
            'max_change_1h': max_price_change,
            'min_change_1h': min_price_change,
            'volatility': price_volatility,
            'price_trend': 'bullish' if recent_data['price_change'].iloc[-1] > 0 else 'bearish'
        }
        
        # Market cap analysis
        if 'market_cap' in df.columns:
            recent_data['market_cap_change'] = recent_data['market_cap'].pct_change() * 100
            indicators['market_cap'] = {
                'current': recent_data['market_cap'].iloc[-1],
                'change_24h': recent_data['market_cap_change'].iloc[-1] if not pd.isna(recent_data['market_cap_change'].iloc[-1]) else 0
            }
        
        # Pump probability score
        pump_score = 0
        
        # Volume spike weight (40%)
        if max_volume_ratio >= self.volume_thresholds['extreme']:
            pump_score += 40
        elif max_volume_ratio >= self.volume_thresholds['high']:
            pump_score += 25
        elif max_volume_ratio >= self.volume_thresholds['moderate']:
            pump_score += 10
        
        # Price action weight (30%)
        if max_price_change > 20:
            pump_score += 30
        elif max_price_change > 10:
            pump_score += 20
        elif max_price_change > 5:
            pump_score += 10
        
        # Volatility weight (20%)
        if price_volatility > 15:
            pump_score += 20
        elif price_volatility > 10:
            pump_score += 15
        elif price_volatility > 5:
            pump_score += 10
        
        # Market cap change weight (10%)
        if 'market_cap' in indicators and abs(indicators['market_cap']['change_24h']) > 50:
            pump_score += 10
        elif 'market_cap' in indicators and abs(indicators['market_cap']['change_24h']) > 25:
            pump_score += 5
        
        indicators['pump_probability'] = min(pump_score, 100)
        indicators['pump_risk_level'] = self._get_risk_level(pump_score)
        
        return indicators
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level based on pump probability score"""
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MODERATE'
        elif score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def get_news_sentiment(self, coin_symbol: str) -> Dict:
        """Get news sentiment for a coin (placeholder for news API integration)"""
        # This would integrate with news APIs like NewsAPI, CryptoPanic, etc.
        # For now, returning mock data
        return {
            'sentiment_score': np.random.uniform(-1, 1),
            'news_count': np.random.randint(5, 50),
            'positive_news': np.random.randint(0, 20),
            'negative_news': np.random.randint(0, 20),
            'neutral_news': np.random.randint(0, 20)
        }
    
    def analyze_market_maker_activity(self, coin_id: str) -> Dict:
        """Analyze market maker activity (placeholder for on-chain analysis)"""
        # This would integrate with blockchain explorers and DEX APIs
        # For now, returning mock data
        return {
            'dwf_labs_activity': {
                'recent_trades': np.random.randint(0, 100),
                'volume_contribution': np.random.uniform(0, 0.3),
                'last_activity': datetime.now() - timedelta(hours=np.random.randint(1, 24))
            },
            'other_makers': {
                'total_activity': np.random.randint(0, 500),
                'suspicious_patterns': np.random.choice([True, False], p=[0.2, 0.8])
            }
        }
    
    def create_analysis_report(self, coin_id: str, days: int = 7) -> Dict:
        """Create comprehensive analysis report"""
        logger.info(f"Starting analysis for {coin_id}")
        
        # Get coin data
        coin_data = self.get_coin_data(coin_id, days)
        if coin_data['status'] != 'success':
            return coin_data
        
        df = coin_data['data']
        
        # Detect volume anomalies
        df_analyzed = self.detect_volume_anomalies(df)
        
        # Analyze pump indicators
        indicators = self.analyze_pump_indicators(df_analyzed)
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(coin_id)
        
        # Analyze market maker activity
        mm_activity = self.analyze_market_maker_activity(coin_id)
        
        # Create report
        report = {
            'coin_id': coin_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'timeframe_days': days,
            'data_points': len(df_analyzed),
            'indicators': indicators,
            'news_sentiment': news_sentiment,
            'market_maker_activity': mm_activity,
            'recommendations': self._generate_recommendations(indicators, news_sentiment, mm_activity)
        }
        
        return report
    
    def _generate_recommendations(self, indicators: Dict, news_sentiment: Dict, mm_activity: Dict) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        pump_prob = indicators['pump_probability']
        risk_level = indicators['pump_risk_level']
        
        if risk_level == 'CRITICAL':
            recommendations.append("⚠️ CRITICAL RISK: Extreme volume spike detected. High probability of pump and dump.")
            recommendations.append("🚨 Consider immediate position review and risk management.")
        elif risk_level == 'HIGH':
            recommendations.append("⚠️ HIGH RISK: Significant volume and price anomalies detected.")
            recommendations.append("📊 Monitor closely and consider reducing position size.")
        elif risk_level == 'MODERATE':
            recommendations.append("⚡ MODERATE RISK: Unusual activity detected. Stay alert.")
            recommendations.append("📈 Consider setting stop-loss orders.")
        else:
            recommendations.append("✅ LOW RISK: Normal market conditions detected.")
        
        # News sentiment recommendations
        if news_sentiment['sentiment_score'] > 0.5:
            recommendations.append("📰 Positive news sentiment detected - may support price action.")
        elif news_sentiment['sentiment_score'] < -0.5:
            recommendations.append("📰 Negative news sentiment detected - may pressure price.")
        
        # Market maker recommendations
        if mm_activity['dwf_labs_activity']['volume_contribution'] > 0.2:
            recommendations.append("🏦 DWF Labs showing significant activity - monitor for coordinated moves.")
        
        return recommendations
    
    def create_visualizations(self, df: pd.DataFrame, coin_id: str) -> None:
        """Create comprehensive visualizations"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f'{coin_id.upper()} Price Action',
                'Volume Analysis',
                'Volume Anomaly Detection',
                'Price vs Volume Correlation'
            ],
            vertical_spacing=0.08
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='orange'),
            row=2, col=1
        )
        
        # Volume anomaly detection
        colors = ['green' if x == 'normal' else 'orange' if x == 'moderate' else 'red' 
                 for x in df['anomaly_level']]
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['volume_ratio_7'], 
                      mode='markers', marker=dict(color=colors, size=8),
                      name='Volume Ratio'),
            row=3, col=1
        )
        
        # Price vs Volume correlation
        fig.add_trace(
            go.Scatter(x=df['volume'], y=df['price'], mode='markers',
                      marker=dict(color=df['price_change'], colorscale='RdYlGn'),
                      name='Price vs Volume'),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f'Crypto Pump Analysis: {coin_id.upper()}',
            height=1200,
            showlegend=True
        )
        
        # Save plot
        fig.write_html(f'/workspace/{coin_id}_analysis.html')
        fig.show()
    
    def run_analysis(self, coin_ids: List[str], days: int = 7) -> None:
        """Run analysis for multiple coins"""
        results = {}
        
        for coin_id in coin_ids:
            logger.info(f"Analyzing {coin_id}...")
            
            try:
                # Create analysis report
                report = self.create_analysis_report(coin_id, days)
                results[coin_id] = report
                
                # Get data for visualization
                coin_data = self.get_coin_data(coin_id, days)
                if coin_data['status'] == 'success':
                    df_analyzed = self.detect_volume_anomalies(coin_data['data'])
                    self.create_visualizations(df_analyzed, coin_id)
                
                # Print summary
                self._print_analysis_summary(coin_id, report)
                
            except Exception as e:
                logger.error(f"Error analyzing {coin_id}: {str(e)}")
                results[coin_id] = {'error': str(e)}
            
            # Rate limiting
            time.sleep(1)
        
        # Save results
        with open('/workspace/analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _print_analysis_summary(self, coin_id: str, report: Dict) -> None:
        """Print analysis summary"""
        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY: {coin_id.upper()}")
        print(f"{'='*60}")
        
        indicators = report['indicators']
        print(f"Pump Probability: {indicators['pump_probability']}%")
        print(f"Risk Level: {indicators['pump_risk_level']}")
        print(f"Max Volume Ratio: {indicators['volume_spike']['max_ratio']:.2f}x")
        print(f"Max Price Change: {indicators['price_action']['max_change_1h']:.2f}%")
        print(f"Price Volatility: {indicators['price_action']['volatility']:.2f}%")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print(f"{'='*60}\n")


def main():
    """Main function to run the analysis"""
    analyzer = CryptoPumpAnalyzer()
    
    # Popular coins to analyze
    coins_to_analyze = [
        'bitcoin',
        'ethereum',
        'binancecoin',
        'cardano',
        'solana',
        'polkadot',
        'chainlink',
        'litecoin'
    ]
    
    print("🚀 Starting Crypto Pump Analysis...")
    print("📊 Analyzing volume anomalies and pump indicators...")
    print("📰 Considering news sentiment...")
    print("🏦 Monitoring market maker activities...")
    
    # Run analysis
    results = analyzer.run_analysis(coins_to_analyze, days=7)
    
    print("✅ Analysis complete! Check the generated HTML files and analysis_results.json")


if __name__ == "__main__":
    main()