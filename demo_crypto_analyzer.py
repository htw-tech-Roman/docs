#!/usr/bin/env python3
"""
Demo Crypto Pump Analyzer
Demonstrates analysis capabilities using simulated data
Focuses on DWF Labs and market maker activities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoCryptoAnalyzer:
    """Demo crypto pump analyzer with simulated data"""
    
    def __init__(self):
        # Market makers to track
        self.market_makers = {
            'DWF Labs': {
                'addresses': [
                    '0x6cc5f688a315f3dc28a7781717a9a798a59fda7b',
                    '0x7f268357a8c2552623316e2562d90e642bb538e5'
                ],
                'risk_level': 'high',
                'known_patterns': ['coordinated_buying', 'volume_spikes', 'price_manipulation']
            },
            'Alameda Research': {
                'addresses': ['0x5f6c97c6ad7bdd0ae7e0dd4ca33a4ed3fd0b4fc'],
                'risk_level': 'critical',
                'known_patterns': ['wash_trading', 'cross_exchange_arbitrage']
            },
            'Jump Trading': {
                'addresses': ['0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf'],
                'risk_level': 'moderate',
                'known_patterns': ['algorithmic_trading', 'high_frequency']
            }
        }
        
        # Volume spike thresholds
        self.volume_thresholds = {
            'extreme': 10.0,  # 10x normal volume
            'high': 5.0,      # 5x normal volume
            'moderate': 2.0   # 2x normal volume
        }
    
    def generate_simulated_data(self, coin_id: str, days: int = 7) -> pd.DataFrame:
        """Generate simulated crypto data with pump patterns"""
        # Generate base data
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                                 end=datetime.now(), freq='H')
        
        # Base price (simulate some coins with different characteristics)
        base_prices = {
            'bitcoin': 45000,
            'ethereum': 3000,
            'solana': 100,
            'cardano': 0.5,
            'chainlink': 15,
            'polkadot': 7
        }
        
        base_price = base_prices.get(coin_id, 100)
        
        # Generate price data with some pump patterns
        np.random.seed(hash(coin_id) % 2**32)  # Consistent seed per coin
        
        # Create some pump events
        pump_events = []
        for i in range(len(timestamps)):
            # Random chance of pump event
            if np.random.random() < 0.05:  # 5% chance per hour
                pump_events.append(i)
        
        # Generate price series
        prices = []
        volumes = []
        market_caps = []
        
        current_price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # Normal price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            
            # Add pump effect if this is a pump event
            if i in pump_events:
                pump_strength = np.random.uniform(0.1, 0.5)  # 10-50% pump
                price_change += pump_strength
                
                # Add volume spike
                volume_multiplier = np.random.uniform(5, 15)  # 5-15x volume
            else:
                volume_multiplier = np.random.uniform(0.5, 2.0)  # Normal volume variation
            
            current_price *= (1 + price_change)
            prices.append(current_price)
            
            # Generate volume (correlated with price movement)
            base_volume = 1000000  # Base volume
            volume = base_volume * volume_multiplier * (1 + abs(price_change) * 2)
            volumes.append(volume)
            
            # Market cap (simplified)
            market_cap = current_price * 1000000  # Assume 1M supply
            market_caps.append(market_cap)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'market_cap': market_caps
        })
        
        return df
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume anomalies using statistical methods"""
        df = df.copy()
        
        # Calculate rolling statistics
        df['volume_ma_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
        df['volume_ma_24'] = df['volume'].rolling(window=24, min_periods=1).mean()
        df['volume_std'] = df['volume'].rolling(window=24, min_periods=1).std()
        
        # Calculate volume ratios
        df['volume_ratio_7'] = df['volume'] / df['volume_ma_7']
        df['volume_ratio_24'] = df['volume'] / df['volume_ma_24']
        
        # Z-score for volume
        df['volume_zscore'] = (df['volume'] - df['volume_ma_24']) / df['volume_std']
        
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
        if abs(indicators['market_cap']['change_24h']) > 50:
            pump_score += 10
        elif abs(indicators['market_cap']['change_24h']) > 25:
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
        """Get news sentiment for a coin (simulated)"""
        # Simulate sentiment data with some bias towards pump mentions
        sentiment_score = np.random.uniform(-0.8, 0.8)
        news_count = np.random.randint(10, 100)
        
        # Simulate pump/dump mentions (biased towards pump in demo)
        pump_mentions = int(news_count * np.random.uniform(0.2, 0.6))
        dump_mentions = int(news_count * np.random.uniform(0.05, 0.3))
        
        return {
            'sentiment_score': sentiment_score,
            'news_count': news_count,
            'positive_news': int(news_count * (0.3 + sentiment_score * 0.2)),
            'negative_news': int(news_count * (0.2 - sentiment_score * 0.2)),
            'neutral_news': news_count - int(news_count * (0.5 + abs(sentiment_score) * 0.2)),
            'pump_mentions': pump_mentions,
            'dump_mentions': dump_mentions,
            'manipulation_risk': np.random.uniform(0, 0.4)
        }
    
    def analyze_dwf_labs_activity(self, coin_id: str) -> Dict:
        """Analyze DWF Labs specific activity (simulated)"""
        # Simulate DWF Labs activity with higher risk patterns
        np.random.seed(hash(coin_id + 'dwf') % 2**32)
        
        # Generate activity data
        activity_data = {
            'total_trades': np.random.randint(50, 500),
            'total_volume': np.random.uniform(1000000, 10000000),
            'buy_volume': np.random.uniform(500000, 8000000),
            'sell_volume': np.random.uniform(300000, 5000000),
            'avg_trade_size': np.random.uniform(10000, 100000),
            'trading_frequency': np.random.uniform(1, 10),  # trades per hour
            'last_activity': datetime.now() - timedelta(hours=np.random.randint(1, 48))
        }
        
        # Analyze patterns
        patterns = []
        risk_score = 0
        
        # Pattern 1: Volume spikes
        if activity_data['total_volume'] > 5000000:
            patterns.append('volume_spikes')
            risk_score += 25
        
        # Pattern 2: High frequency trading
        if activity_data['trading_frequency'] > 5:
            patterns.append('high_frequency_trading')
            risk_score += 20
        
        # Pattern 3: Coordinated buying
        buy_ratio = activity_data['buy_volume'] / activity_data['total_volume']
        if buy_ratio > 0.7:
            patterns.append('coordinated_buying')
            risk_score += 30
        
        # Pattern 4: Large trades
        if activity_data['avg_trade_size'] > 50000:
            patterns.append('large_trades')
            risk_score += 15
        
        # Pattern 5: Recent activity
        hours_since_activity = (datetime.now() - activity_data['last_activity']).total_seconds() / 3600
        if hours_since_activity < 6:
            patterns.append('recent_activity')
            risk_score += 10
        
        return {
            'market_maker': 'DWF Labs',
            'activity_data': activity_data,
            'patterns_detected': patterns,
            'risk_score': min(risk_score, 100),
            'risk_level': self._get_risk_level(risk_score),
            'alerts': self._generate_dwf_alerts(patterns, risk_score)
        }
    
    def _generate_dwf_alerts(self, patterns: List[str], risk_score: int) -> List[Dict]:
        """Generate alerts for DWF Labs activity"""
        alerts = []
        
        if risk_score >= 80:
            alerts.append({
                'type': 'critical',
                'message': 'DWF Labs showing critical risk activity',
                'patterns': patterns
            })
        elif risk_score >= 60:
            alerts.append({
                'type': 'high',
                'message': 'DWF Labs showing high risk patterns',
                'patterns': patterns
            })
        
        if 'coordinated_buying' in patterns:
            alerts.append({
                'type': 'pattern',
                'message': 'Coordinated buying patterns detected',
                'severity': 'high'
            })
        
        if 'volume_spikes' in patterns:
            alerts.append({
                'type': 'pattern',
                'message': 'Unusual volume spikes detected',
                'severity': 'moderate'
            })
        
        return alerts
    
    def create_analysis_report(self, coin_id: str, days: int = 7) -> Dict:
        """Create comprehensive analysis report"""
        logger.info(f"Starting analysis for {coin_id}")
        
        # Generate simulated data
        df = self.generate_simulated_data(coin_id, days)
        
        # Detect volume anomalies
        df_analyzed = self.detect_volume_anomalies(df)
        
        # Analyze pump indicators
        indicators = self.analyze_pump_indicators(df_analyzed)
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(coin_id)
        
        # Analyze DWF Labs activity
        dwf_analysis = self.analyze_dwf_labs_activity(coin_id)
        
        # Calculate comprehensive risk score
        volume_risk = indicators['pump_probability']
        sentiment_risk = abs(news_sentiment['sentiment_score']) * 50 + 25
        dwf_risk = dwf_analysis['risk_score']
        
        # Weighted average
        comprehensive_risk = (volume_risk * 0.4 + sentiment_risk * 0.3 + dwf_risk * 0.3)
        
        # Create report
        report = {
            'coin_id': coin_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'timeframe_days': days,
            'data_points': len(df_analyzed),
            'indicators': indicators,
            'news_sentiment': news_sentiment,
            'dwf_labs_analysis': dwf_analysis,
            'comprehensive_risk_score': comprehensive_risk,
            'comprehensive_risk_level': self._get_risk_level(comprehensive_risk),
            'recommendations': self._generate_recommendations(indicators, news_sentiment, dwf_analysis, comprehensive_risk)
        }
        
        return report
    
    def _generate_recommendations(self, indicators: Dict, news_sentiment: Dict, dwf_analysis: Dict, risk_score: float) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        risk_level = self._get_risk_level(risk_score)
        
        if risk_level == 'CRITICAL':
            recommendations.append("🚨 CRITICAL RISK: Multiple indicators suggest high probability of pump and dump")
            recommendations.append("🚨 Consider immediate position review and risk management")
        elif risk_level == 'HIGH':
            recommendations.append("⚠️ HIGH RISK: Strong indicators of potential manipulation")
            recommendations.append("📊 Monitor closely and consider reducing position size")
        elif risk_level == 'MODERATE':
            recommendations.append("⚡ MODERATE RISK: Unusual activity detected. Stay alert")
            recommendations.append("📈 Consider setting stop-loss orders")
        else:
            recommendations.append("✅ LOW RISK: Normal market conditions detected")
        
        # DWF Labs specific recommendations
        if dwf_analysis['risk_score'] > 70:
            recommendations.append("🏦 DWF Labs showing high risk activity - monitor for coordinated moves")
        
        if 'coordinated_buying' in dwf_analysis['patterns_detected']:
            recommendations.append("⚠️ Coordinated buying patterns detected - high manipulation risk")
        
        # News sentiment recommendations
        if news_sentiment['manipulation_risk'] > 0.3:
            recommendations.append("📰 High manipulation risk in social sentiment - be skeptical of hype")
        
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
                
                # Generate data for visualization
                df = self.generate_simulated_data(coin_id, days)
                df_analyzed = self.detect_volume_anomalies(df)
                self.create_visualizations(df_analyzed, coin_id)
                
                # Print summary
                self._print_analysis_summary(coin_id, report)
                
            except Exception as e:
                logger.error(f"Error analyzing {coin_id}: {str(e)}")
                results[coin_id] = {'error': str(e)}
        
        # Save results
        with open('/workspace/demo_analysis_results.json', 'w') as f:
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
        
        print(f"\nDWF Labs Analysis:")
        dwf = report['dwf_labs_analysis']
        print(f"  Risk Score: {dwf['risk_score']}")
        print(f"  Risk Level: {dwf['risk_level']}")
        print(f"  Patterns: {', '.join(dwf['patterns_detected'])}")
        print(f"  Alerts: {len(dwf['alerts'])}")
        
        print(f"\nNews Sentiment:")
        news = report['news_sentiment']
        print(f"  Sentiment Score: {news['sentiment_score']:.3f}")
        print(f"  Pump Mentions: {news['pump_mentions']}")
        print(f"  Manipulation Risk: {news['manipulation_risk']:.1%}")
        
        print(f"\nComprehensive Risk Score: {report['comprehensive_risk_score']:.1f}")
        print(f"Comprehensive Risk Level: {report['comprehensive_risk_level']}")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print(f"{'='*60}\n")


def main():
    """Main function to run the demo analysis"""
    analyzer = DemoCryptoAnalyzer()
    
    # Popular coins to analyze
    coins_to_analyze = [
        'bitcoin',
        'ethereum',
        'solana',
        'cardano',
        'chainlink',
        'polkadot'
    ]
    
    print("🚀 Starting Demo Crypto Pump Analysis...")
    print("📊 Analyzing volume anomalies and pump indicators...")
    print("📰 Considering news sentiment...")
    print("🏦 Monitoring DWF Labs and other market maker activities...")
    print("🎯 Focusing on pump and dump detection...")
    
    # Run analysis
    results = analyzer.run_analysis(coins_to_analyze, days=7)
    
    # Create summary
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    risk_levels = {}
    dwf_high_risk = 0
    
    for coin_id, analysis in results.items():
        if 'error' not in analysis:
            risk_level = analysis['comprehensive_risk_level']
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            
            if analysis['dwf_labs_analysis']['risk_score'] > 70:
                dwf_high_risk += 1
    
    print("Risk Distribution:")
    for level, count in risk_levels.items():
        print(f"  {level}: {count}")
    
    print(f"\nDWF Labs High Risk Coins: {dwf_high_risk}")
    
    # Show highest risk coins
    high_risk_coins = []
    for coin_id, analysis in results.items():
        if 'error' not in analysis:
            high_risk_coins.append((coin_id, analysis['comprehensive_risk_score']))
    
    high_risk_coins.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nHighest Risk Coins:")
    for i, (coin_id, score) in enumerate(high_risk_coins[:5], 1):
        print(f"  {i}. {coin_id}: {score:.1f}")
    
    print("\n✅ Demo analysis complete! Check the generated HTML files and demo_analysis_results.json")
    print("📈 This demonstrates the analysis capabilities with simulated data")
    print("🔧 In production, this would use real data from CoinGecko, news APIs, and blockchain explorers")


if __name__ == "__main__":
    main()