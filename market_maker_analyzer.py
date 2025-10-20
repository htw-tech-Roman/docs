#!/usr/bin/env python3
"""
Market Maker Activity Analyzer
Focuses on DWF Labs and other major market makers
Analyzes their trading patterns and potential pump activities
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class MarketMakerAnalyzer:
    """Analyzes market maker activities and patterns"""
    
    def __init__(self):
        self.market_makers = {
            'DWF Labs': {
                'addresses': [
                    '0x6cc5f688a315f3dc28a7781717a9a798a59fda7b',  # DWF Labs main
                    '0x7f268357a8c2552623316e2562d90e642bb538e5',  # DWF Labs trading
                ],
                'known_patterns': ['coordinated_buying', 'volume_spikes', 'price_manipulation'],
                'risk_level': 'high'
            },
            'Alameda Research': {
                'addresses': [
                    '0x5f6c97c6ad7bdd0ae7e0dd4ca33a4ed3fd0b4fc',
                    '0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a'
                ],
                'known_patterns': ['wash_trading', 'cross_exchange_arbitrage'],
                'risk_level': 'critical'
            },
            'Jump Trading': {
                'addresses': [
                    '0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf',
                    '0x26aad2da94c59524ac0d93f6d6cbf9071d7086f2'
                ],
                'known_patterns': ['algorithmic_trading', 'high_frequency'],
                'risk_level': 'moderate'
            },
            'Wintermute': {
                'addresses': [
                    '0x0000000000000000000000000000000000000000',  # Placeholder
                ],
                'known_patterns': ['market_making', 'arbitrage'],
                'risk_level': 'moderate'
            },
            'GSR': {
                'addresses': [
                    '0x0000000000000000000000000000000000000000',  # Placeholder
                ],
                'known_patterns': ['institutional_trading'],
                'risk_level': 'low'
            }
        }
        
        # Exchange addresses for tracking
        self.exchanges = {
            'binance': '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be',
            'coinbase': '0x71660c4005ba85c37ccec55d0c4493e66fe775d3',
            'kraken': '0x2910543af39aba0cd09dbb2d50200b3e800a63d2',
            'okx': '0x6cc5f688a315f3dc28a7781717a9a798a59fda7b'
        }
        
        # Suspicious trading patterns
        self.suspicious_patterns = {
            'wash_trading': {
                'description': 'Trading between controlled addresses',
                'indicators': ['circular_trades', 'same_amount_trades', 'rapid_buy_sell']
            },
            'pump_and_dump': {
                'description': 'Coordinated price manipulation',
                'indicators': ['volume_spike', 'price_spike', 'coordinated_timing']
            },
            'spoofing': {
                'description': 'Large orders to manipulate price',
                'indicators': ['large_cancelled_orders', 'price_impact', 'order_book_manipulation']
            },
            'layering': {
                'description': 'Multiple orders to create false depth',
                'indicators': ['multiple_small_orders', 'order_cancellation', 'depth_manipulation']
            }
        }
    
    def analyze_dwf_labs_activity(self, coin_address: str, days: int = 7) -> Dict:
        """Analyze DWF Labs specific activity"""
        logger.info(f"Analyzing DWF Labs activity for {coin_address}")
        
        # Simulate DWF Labs activity data
        # In practice, this would query blockchain explorers and DEX APIs
        dwf_data = self._simulate_dwf_activity(coin_address, days)
        
        # Analyze patterns
        patterns = self._analyze_trading_patterns(dwf_data)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(dwf_data, patterns)
        
        # Generate alerts
        alerts = self._generate_dwf_alerts(dwf_data, patterns, risk_metrics)
        
        return {
            'market_maker': 'DWF Labs',
            'coin_address': coin_address,
            'analysis_period': f"{days} days",
            'timestamp': datetime.now().isoformat(),
            'activity_data': dwf_data,
            'patterns_detected': patterns,
            'risk_metrics': risk_metrics,
            'alerts': alerts,
            'recommendations': self._generate_dwf_recommendations(patterns, risk_metrics)
        }
    
    def _simulate_dwf_activity(self, coin_address: str, days: int) -> Dict:
        """Simulate DWF Labs activity data"""
        # This would be replaced with real blockchain data
        now = datetime.now()
        
        # Generate trading activity
        trades = []
        for i in range(days * 24):  # Hourly data
            timestamp = now - timedelta(hours=i)
            
            # Simulate DWF Labs trading patterns
            if np.random.random() < 0.3:  # 30% chance of activity per hour
                trade = {
                    'timestamp': timestamp,
                    'type': np.random.choice(['buy', 'sell']),
                    'amount': np.random.exponential(100000),  # Exponential distribution for trade sizes
                    'price': np.random.normal(1.0, 0.05),  # Price around $1 with 5% volatility
                    'volume': np.random.exponential(50000),
                    'gas_used': np.random.randint(50000, 200000),
                    'transaction_hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}"
                }
                trades.append(trade)
        
        # Calculate aggregated metrics
        total_volume = sum(trade['volume'] for trade in trades)
        buy_volume = sum(trade['volume'] for trade in trades if trade['type'] == 'buy')
        sell_volume = sum(trade['volume'] for trade in trades if trade['type'] == 'sell')
        
        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_volume': buy_volume - sell_volume,
            'avg_trade_size': total_volume / len(trades) if trades else 0,
            'trading_frequency': len(trades) / (days * 24)  # Trades per hour
        }
    
    def _analyze_trading_patterns(self, activity_data: Dict) -> Dict:
        """Analyze trading patterns for suspicious activity"""
        trades = activity_data['trades']
        
        if not trades:
            return {'patterns': [], 'suspicious_score': 0}
        
        patterns = []
        suspicious_score = 0
        
        # Pattern 1: Rapid buy/sell cycles (potential wash trading)
        buy_sell_cycles = 0
        for i in range(len(trades) - 1):
            if trades[i]['type'] != trades[i+1]['type']:
                time_diff = (trades[i+1]['timestamp'] - trades[i]['timestamp']).total_seconds()
                if time_diff < 300:  # Less than 5 minutes
                    buy_sell_cycles += 1
        
        if buy_sell_cycles > len(trades) * 0.3:
            patterns.append('rapid_buy_sell_cycles')
            suspicious_score += 30
        
        # Pattern 2: Similar trade amounts (potential coordinated trading)
        trade_amounts = [trade['amount'] for trade in trades]
        if len(trade_amounts) > 5:
            amount_std = np.std(trade_amounts)
            amount_mean = np.mean(trade_amounts)
            cv = amount_std / amount_mean if amount_mean > 0 else 0
            
            if cv < 0.1:  # Low coefficient of variation
                patterns.append('similar_trade_amounts')
                suspicious_score += 25
        
        # Pattern 3: High trading frequency
        if activity_data['trading_frequency'] > 2:  # More than 2 trades per hour
            patterns.append('high_frequency_trading')
            suspicious_score += 15
        
        # Pattern 4: Large volume spikes
        hourly_volumes = {}
        for trade in trades:
            hour_key = trade['timestamp'].replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_volumes:
                hourly_volumes[hour_key] = 0
            hourly_volumes[hour_key] += trade['volume']
        
        if hourly_volumes:
            volumes = list(hourly_volumes.values())
            max_volume = max(volumes)
            avg_volume = np.mean(volumes)
            
            if max_volume > avg_volume * 5:  # 5x average volume
                patterns.append('volume_spikes')
                suspicious_score += 35
        
        # Pattern 5: Price manipulation indicators
        prices = [trade['price'] for trade in trades]
        if len(prices) > 10:
            price_changes = np.diff(prices)
            large_price_changes = np.sum(np.abs(price_changes) > np.std(price_changes) * 2)
            
            if large_price_changes > len(price_changes) * 0.2:
                patterns.append('price_manipulation')
                suspicious_score += 40
        
        return {
            'patterns': patterns,
            'suspicious_score': min(suspicious_score, 100),
            'pattern_details': {
                'buy_sell_cycles': buy_sell_cycles,
                'trading_frequency': activity_data['trading_frequency'],
                'volume_spike_ratio': max(volumes) / np.mean(volumes) if hourly_volumes else 1
            }
        }
    
    def _calculate_risk_metrics(self, activity_data: Dict, patterns: Dict) -> Dict:
        """Calculate risk metrics for market maker activity"""
        trades = activity_data['trades']
        
        if not trades:
            return {'risk_level': 'unknown', 'risk_score': 0}
        
        risk_score = patterns['suspicious_score']
        
        # Additional risk factors
        volume_risk = 0
        if activity_data['total_volume'] > 1000000:  # High volume threshold
            volume_risk = 20
        
        frequency_risk = 0
        if activity_data['trading_frequency'] > 5:  # Very high frequency
            frequency_risk = 15
        
        net_position_risk = 0
        net_ratio = abs(activity_data['net_volume']) / activity_data['total_volume']
        if net_ratio > 0.8:  # Highly directional
            net_position_risk = 10
        
        total_risk = risk_score + volume_risk + frequency_risk + net_position_risk
        
        # Determine risk level
        if total_risk >= 80:
            risk_level = 'CRITICAL'
        elif total_risk >= 60:
            risk_level = 'HIGH'
        elif total_risk >= 40:
            risk_level = 'MODERATE'
        elif total_risk >= 20:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'risk_level': risk_level,
            'risk_score': min(total_risk, 100),
            'volume_risk': volume_risk,
            'frequency_risk': frequency_risk,
            'net_position_risk': net_position_risk,
            'suspicious_patterns': len(patterns['patterns']),
            'total_trades': activity_data['total_trades'],
            'total_volume': activity_data['total_volume']
        }
    
    def _generate_dwf_alerts(self, activity_data: Dict, patterns: Dict, risk_metrics: Dict) -> List[Dict]:
        """Generate alerts for DWF Labs activity"""
        alerts = []
        
        # High risk alert
        if risk_metrics['risk_level'] in ['CRITICAL', 'HIGH']:
            alerts.append({
                'type': 'risk_alert',
                'severity': risk_metrics['risk_level'],
                'message': f"DWF Labs showing {risk_metrics['risk_level']} risk activity",
                'timestamp': datetime.now().isoformat()
            })
        
        # Pattern alerts
        for pattern in patterns['patterns']:
            if pattern == 'rapid_buy_sell_cycles':
                alerts.append({
                    'type': 'pattern_alert',
                    'severity': 'HIGH',
                    'message': 'Rapid buy/sell cycles detected - potential wash trading',
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat()
                })
            elif pattern == 'volume_spikes':
                alerts.append({
                    'type': 'pattern_alert',
                    'severity': 'MODERATE',
                    'message': 'Unusual volume spikes detected',
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat()
                })
            elif pattern == 'price_manipulation':
                alerts.append({
                    'type': 'pattern_alert',
                    'severity': 'CRITICAL',
                    'message': 'Price manipulation patterns detected',
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Volume alert
        if activity_data['total_volume'] > 5000000:  # $5M+ volume
            alerts.append({
                'type': 'volume_alert',
                'severity': 'MODERATE',
                'message': f"High volume activity: ${activity_data['total_volume']:,.0f}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _generate_dwf_recommendations(self, patterns: Dict, risk_metrics: Dict) -> List[str]:
        """Generate recommendations based on DWF Labs analysis"""
        recommendations = []
        
        risk_level = risk_metrics['risk_level']
        
        if risk_level == 'CRITICAL':
            recommendations.append("🚨 CRITICAL: DWF Labs activity shows high manipulation risk")
            recommendations.append("⚠️ Consider immediate position review and risk management")
            recommendations.append("📊 Monitor for potential pump and dump activity")
        elif risk_level == 'HIGH':
            recommendations.append("⚠️ HIGH RISK: Suspicious DWF Labs patterns detected")
            recommendations.append("📈 Monitor closely for coordinated moves")
            recommendations.append("🛡️ Consider setting tighter stop-loss orders")
        elif risk_level == 'MODERATE':
            recommendations.append("⚡ MODERATE RISK: Unusual DWF Labs activity")
            recommendations.append("👀 Stay alert for potential manipulation")
        else:
            recommendations.append("✅ LOW RISK: Normal DWF Labs market making activity")
        
        # Pattern-specific recommendations
        if 'rapid_buy_sell_cycles' in patterns['patterns']:
            recommendations.append("🔄 Wash trading patterns detected - be cautious")
        
        if 'volume_spikes' in patterns['patterns']:
            recommendations.append("📊 Volume spikes detected - monitor for pump activity")
        
        if 'price_manipulation' in patterns['patterns']:
            recommendations.append("💰 Price manipulation detected - high risk of coordinated moves")
        
        return recommendations
    
    def analyze_all_market_makers(self, coin_address: str, days: int = 7) -> Dict:
        """Analyze all known market makers for a coin"""
        logger.info(f"Analyzing all market makers for {coin_address}")
        
        all_analyses = {}
        
        for mm_name, mm_data in self.market_makers.items():
            if mm_name == 'DWF Labs':
                analysis = self.analyze_dwf_labs_activity(coin_address, days)
            else:
                # Simulate analysis for other market makers
                analysis = self._simulate_mm_analysis(mm_name, coin_address, days)
            
            all_analyses[mm_name] = analysis
        
        # Calculate overall market maker risk
        risk_scores = [analysis['risk_metrics']['risk_score'] for analysis in all_analyses.values()]
        overall_risk = np.mean(risk_scores) if risk_scores else 0
        
        # Count critical/high risk makers
        critical_makers = [name for name, analysis in all_analyses.items() 
                          if analysis['risk_metrics']['risk_level'] in ['CRITICAL', 'HIGH']]
        
        return {
            'coin_address': coin_address,
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_risk_score': overall_risk,
            'critical_makers': critical_makers,
            'market_maker_analyses': all_analyses,
            'summary': {
                'total_makers_analyzed': len(all_analyses),
                'high_risk_makers': len(critical_makers),
                'risk_distribution': {
                    'critical': len([a for a in all_analyses.values() if a['risk_metrics']['risk_level'] == 'CRITICAL']),
                    'high': len([a for a in all_analyses.values() if a['risk_metrics']['risk_level'] == 'HIGH']),
                    'moderate': len([a for a in all_analyses.values() if a['risk_metrics']['risk_level'] == 'MODERATE']),
                    'low': len([a for a in all_analyses.values() if a['risk_metrics']['risk_level'] == 'LOW'])
                }
            }
        }
    
    def _simulate_mm_analysis(self, mm_name: str, coin_address: str, days: int) -> Dict:
        """Simulate analysis for other market makers"""
        # Generate random but realistic data
        risk_levels = ['MINIMAL', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        risk_weights = [0.3, 0.3, 0.2, 0.15, 0.05]  # Most are low risk
        
        risk_level = np.random.choice(risk_levels, p=risk_weights)
        risk_score = {'MINIMAL': 10, 'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'CRITICAL': 90}[risk_level]
        
        return {
            'market_maker': mm_name,
            'coin_address': coin_address,
            'analysis_period': f"{days} days",
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'total_trades': np.random.randint(10, 1000),
                'total_volume': np.random.exponential(1000000)
            },
            'patterns_detected': np.random.choice([[], ['volume_spikes'], ['high_frequency_trading']], p=[0.7, 0.2, 0.1]),
            'alerts': [],
            'recommendations': [f"Normal {mm_name} activity detected"]
        }
    
    def create_market_maker_dashboard(self, analysis_results: Dict) -> None:
        """Create a dashboard for market maker analysis"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Market Maker Risk Distribution',
                'Risk Scores by Market Maker',
                'Trading Volume by Market Maker',
                'Alert Summary'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Risk distribution pie chart
        risk_dist = analysis_results['summary']['risk_distribution']
        fig.add_trace(
            go.Pie(
                labels=list(risk_dist.keys()),
                values=list(risk_dist.values()),
                name="Risk Distribution"
            ),
            row=1, col=1
        )
        
        # Risk scores bar chart
        mm_names = list(analysis_results['market_maker_analyses'].keys())
        risk_scores = [analysis_results['market_maker_analyses'][name]['risk_metrics']['risk_score'] 
                      for name in mm_names]
        
        fig.add_trace(
            go.Bar(x=mm_names, y=risk_scores, name="Risk Scores"),
            row=1, col=2
        )
        
        # Trading volume
        volumes = [analysis_results['market_maker_analyses'][name]['risk_metrics']['total_volume'] 
                  for name in mm_names]
        
        fig.add_trace(
            go.Bar(x=mm_names, y=volumes, name="Trading Volume"),
            row=2, col=1
        )
        
        # Alert summary
        alert_counts = {}
        for mm_name, analysis in analysis_results['market_maker_analyses'].items():
            alert_counts[mm_name] = len(analysis['alerts'])
        
        fig.add_trace(
            go.Bar(x=list(alert_counts.keys()), y=list(alert_counts.values()), name="Alerts"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Market Maker Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html('/workspace/market_maker_dashboard.html')
        fig.show()


def main():
    """Test the market maker analyzer"""
    analyzer = MarketMakerAnalyzer()
    
    # Test with a sample coin
    coin_address = "0x1234567890123456789012345678901234567890"
    
    print("🏦 Analyzing DWF Labs activity...")
    dwf_analysis = analyzer.analyze_dwf_labs_activity(coin_address)
    
    print(f"\nDWF Labs Analysis Results:")
    print(f"Risk Level: {dwf_analysis['risk_metrics']['risk_level']}")
    print(f"Risk Score: {dwf_analysis['risk_metrics']['risk_score']}")
    print(f"Patterns Detected: {dwf_analysis['patterns_detected']['patterns']}")
    print(f"Alerts: {len(dwf_analysis['alerts'])}")
    
    print(f"\nRecommendations:")
    for rec in dwf_analysis['recommendations']:
        print(f"  {rec}")
    
    print(f"\n🏦 Analyzing all market makers...")
    all_analysis = analyzer.analyze_all_market_makers(coin_address)
    
    print(f"\nOverall Market Maker Risk: {all_analysis['overall_risk_score']:.1f}")
    print(f"Critical/High Risk Makers: {len(all_analysis['critical_makers'])}")


if __name__ == "__main__":
    main()