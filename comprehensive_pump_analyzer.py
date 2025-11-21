#!/usr/bin/env python3
"""
Comprehensive Crypto Pump Analyzer
Integrates volume analysis, news sentiment, and market maker activity
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from crypto_pump_analyzer import CryptoPumpAnalyzer
from news_sentiment_analyzer import NewsSentimentAnalyzer
from market_maker_analyzer import MarketMakerAnalyzer

logger = logging.getLogger(__name__)

class ComprehensivePumpAnalyzer:
    """Main class that integrates all analysis components"""
    
    def __init__(self):
        self.volume_analyzer = CryptoPumpAnalyzer()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.mm_analyzer = MarketMakerAnalyzer()
        
        # Analysis weights for final scoring
        self.weights = {
            'volume_analysis': 0.4,      # 40% weight
            'sentiment_analysis': 0.3,   # 30% weight
            'market_maker_analysis': 0.3  # 30% weight
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'CRITICAL': 80,
            'HIGH': 60,
            'MODERATE': 40,
            'LOW': 20,
            'MINIMAL': 0
        }
    
    def analyze_coin(self, coin_id: str, coin_symbol: str, coin_address: str = None, days: int = 7) -> Dict:
        """Perform comprehensive analysis for a single coin"""
        logger.info(f"Starting comprehensive analysis for {coin_id} ({coin_symbol})")
        
        analysis_results = {
            'coin_id': coin_id,
            'coin_symbol': coin_symbol,
            'coin_address': coin_address,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_period_days': days
        }
        
        try:
            # 1. Volume and price analysis
            logger.info("📊 Analyzing volume and price patterns...")
            volume_analysis = self.volume_analyzer.create_analysis_report(coin_id, days)
            analysis_results['volume_analysis'] = volume_analysis
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {str(e)}")
            analysis_results['volume_analysis'] = {'error': str(e)}
        
        try:
            # 2. News sentiment analysis
            logger.info("📰 Analyzing news sentiment...")
            sentiment_analysis = self.sentiment_analyzer.get_comprehensive_sentiment(coin_symbol)
            analysis_results['sentiment_analysis'] = sentiment_analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            analysis_results['sentiment_analysis'] = {'error': str(e)}
        
        try:
            # 3. Market maker analysis (if address provided)
            if coin_address:
                logger.info("🏦 Analyzing market maker activity...")
                mm_analysis = self.mm_analyzer.analyze_all_market_makers(coin_address, days)
                analysis_results['market_maker_analysis'] = mm_analysis
            else:
                logger.warning("No coin address provided - skipping market maker analysis")
                analysis_results['market_maker_analysis'] = {'skipped': 'No address provided'}
                
        except Exception as e:
            logger.error(f"Market maker analysis failed: {str(e)}")
            analysis_results['market_maker_analysis'] = {'error': str(e)}
        
        # 4. Calculate comprehensive risk score
        comprehensive_score = self._calculate_comprehensive_score(analysis_results)
        analysis_results['comprehensive_analysis'] = comprehensive_score
        
        # 5. Generate final recommendations
        recommendations = self._generate_final_recommendations(analysis_results)
        analysis_results['final_recommendations'] = recommendations
        
        return analysis_results
    
    def _calculate_comprehensive_score(self, analysis_results: Dict) -> Dict:
        """Calculate comprehensive risk score from all analyses"""
        scores = {}
        weights = self.weights
        
        # Volume analysis score
        if 'volume_analysis' in analysis_results and 'error' not in analysis_results['volume_analysis']:
            volume_score = analysis_results['volume_analysis']['indicators']['pump_probability']
            scores['volume_score'] = volume_score
        else:
            scores['volume_score'] = 0
            weights['volume_analysis'] = 0
        
        # Sentiment analysis score
        if 'sentiment_analysis' in analysis_results and 'error' not in analysis_results['sentiment_analysis']:
            sentiment = analysis_results['sentiment_analysis']['overall_sentiment']
            # Convert sentiment (-1 to 1) to risk score (0 to 100)
            # Extreme sentiment (positive or negative) indicates higher risk
            sentiment_risk = abs(sentiment) * 50 + 25  # 25-75 range
            manipulation_risk = analysis_results['sentiment_analysis']['manipulation_risk'] * 100
            sentiment_score = max(sentiment_risk, manipulation_risk)
            scores['sentiment_score'] = sentiment_score
        else:
            scores['sentiment_score'] = 0
            weights['sentiment_analysis'] = 0
        
        # Market maker analysis score
        if 'market_maker_analysis' in analysis_results and 'error' not in analysis_results['market_maker_analysis']:
            mm_score = analysis_results['market_maker_analysis']['overall_risk_score']
            scores['market_maker_score'] = mm_score
        else:
            scores['market_maker_score'] = 0
            weights['market_maker_analysis'] = 0
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_score = (
                scores['volume_score'] * weights['volume_analysis'] +
                scores['sentiment_score'] * weights['sentiment_analysis'] +
                scores['market_maker_score'] * weights['market_maker_analysis']
            ) / total_weight
        else:
            weighted_score = 0
        
        # Determine risk level
        risk_level = self._get_risk_level(weighted_score)
        
        # Calculate confidence based on available data
        available_analyses = sum(1 for key in ['volume_analysis', 'sentiment_analysis', 'market_maker_analysis']
                               if key in analysis_results and 'error' not in analysis_results[key])
        confidence = available_analyses / 3.0
        
        return {
            'comprehensive_risk_score': weighted_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'component_scores': scores,
            'weights_used': weights,
            'available_analyses': available_analyses
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on comprehensive score"""
        if score >= self.risk_thresholds['CRITICAL']:
            return 'CRITICAL'
        elif score >= self.risk_thresholds['HIGH']:
            return 'HIGH'
        elif score >= self.risk_thresholds['MODERATE']:
            return 'MODERATE'
        elif score >= self.risk_thresholds['LOW']:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_final_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate final recommendations based on comprehensive analysis"""
        recommendations = []
        comprehensive = analysis_results['comprehensive_analysis']
        risk_level = comprehensive['risk_level']
        risk_score = comprehensive['comprehensive_risk_score']
        
        # Risk level recommendations
        if risk_level == 'CRITICAL':
            recommendations.append({
                'type': 'risk_management',
                'priority': 'CRITICAL',
                'message': '🚨 CRITICAL RISK: Multiple indicators suggest high probability of pump and dump',
                'actions': [
                    'Immediately review all positions',
                    'Consider reducing position size by 50-80%',
                    'Set tight stop-loss orders',
                    'Monitor for coordinated selling'
                ]
            })
        elif risk_level == 'HIGH':
            recommendations.append({
                'type': 'risk_management',
                'priority': 'HIGH',
                'message': '⚠️ HIGH RISK: Strong indicators of potential manipulation',
                'actions': [
                    'Reduce position size by 30-50%',
                    'Set stop-loss orders',
                    'Monitor market maker activity closely',
                    'Be prepared for high volatility'
                ]
            })
        elif risk_level == 'MODERATE':
            recommendations.append({
                'type': 'risk_management',
                'priority': 'MODERATE',
                'message': '⚡ MODERATE RISK: Some unusual activity detected',
                'actions': [
                    'Monitor closely',
                    'Consider setting stop-loss orders',
                    'Be cautious with new positions'
                ]
            })
        else:
            recommendations.append({
                'type': 'risk_management',
                'priority': 'LOW',
                'message': '✅ LOW RISK: Normal market conditions',
                'actions': [
                    'Continue normal trading strategy',
                    'Monitor for changes in conditions'
                ]
            })
        
        # Component-specific recommendations
        if 'volume_analysis' in analysis_results and 'error' not in analysis_results['volume_analysis']:
            volume_indicators = analysis_results['volume_analysis']['indicators']
            if volume_indicators['volume_spike']['max_ratio'] > 5:
                recommendations.append({
                    'type': 'volume_alert',
                    'priority': 'HIGH',
                    'message': f"📊 Extreme volume spike detected: {volume_indicators['volume_spike']['max_ratio']:.1f}x normal",
                    'actions': ['Monitor for price manipulation', 'Be cautious of coordinated moves']
                })
        
        if 'sentiment_analysis' in analysis_results and 'error' not in analysis_results['sentiment_analysis']:
            sentiment = analysis_results['sentiment_analysis']
            if sentiment['manipulation_risk'] > 0.3:
                recommendations.append({
                    'type': 'sentiment_alert',
                    'priority': 'MODERATE',
                    'message': f"📰 High manipulation risk in social sentiment: {sentiment['manipulation_risk']:.1%}",
                    'actions': ['Be skeptical of social media hype', 'Verify information independently']
                })
        
        if 'market_maker_analysis' in analysis_results and 'error' not in analysis_results['market_maker_analysis']:
            mm_analysis = analysis_results['market_maker_analysis']
            if len(mm_analysis['critical_makers']) > 0:
                recommendations.append({
                    'type': 'market_maker_alert',
                    'priority': 'HIGH',
                    'message': f"🏦 Critical risk market makers active: {', '.join(mm_analysis['critical_makers'])}",
                    'actions': ['Monitor for coordinated trading', 'Be extra cautious with large positions']
                })
        
        # Confidence-based recommendations
        if comprehensive['confidence'] < 0.5:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'MODERATE',
                'message': f"⚠️ Low confidence analysis ({comprehensive['confidence']:.1%}) - limited data available",
                'actions': ['Gather more data before making decisions', 'Use additional analysis methods']
            })
        
        return recommendations
    
    def analyze_multiple_coins(self, coins: List[Dict], days: int = 7) -> Dict:
        """Analyze multiple coins and compare results"""
        logger.info(f"Starting comprehensive analysis for {len(coins)} coins")
        
        results = {}
        risk_summary = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MODERATE': 0,
            'LOW': 0,
            'MINIMAL': 0
        }
        
        for coin in coins:
            coin_id = coin.get('coin_id')
            coin_symbol = coin.get('coin_symbol', coin_id)
            coin_address = coin.get('coin_address')
            
            logger.info(f"Analyzing {coin_id}...")
            
            try:
                analysis = self.analyze_coin(coin_id, coin_symbol, coin_address, days)
                results[coin_id] = analysis
                
                # Update risk summary
                risk_level = analysis['comprehensive_analysis']['risk_level']
                risk_summary[risk_level] += 1
                
            except Exception as e:
                logger.error(f"Failed to analyze {coin_id}: {str(e)}")
                results[coin_id] = {'error': str(e)}
        
        # Create summary
        summary = {
            'total_coins_analyzed': len(coins),
            'successful_analyses': len([r for r in results.values() if 'error' not in r]),
            'failed_analyses': len([r for r in results.values() if 'error' in r]),
            'risk_distribution': risk_summary,
            'highest_risk_coins': self._get_highest_risk_coins(results),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return {
            'summary': summary,
            'coin_analyses': results
        }
    
    def _get_highest_risk_coins(self, results: Dict) -> List[Dict]:
        """Get coins with highest risk scores"""
        risk_scores = []
        
        for coin_id, analysis in results.items():
            if 'error' not in analysis and 'comprehensive_analysis' in analysis:
                risk_scores.append({
                    'coin_id': coin_id,
                    'risk_score': analysis['comprehensive_analysis']['comprehensive_risk_score'],
                    'risk_level': analysis['comprehensive_analysis']['risk_level']
                })
        
        # Sort by risk score (descending)
        risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return risk_scores[:10]  # Top 10 highest risk
    
    def create_comprehensive_dashboard(self, analysis_results: Dict) -> None:
        """Create comprehensive dashboard for analysis results"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Extract data for visualization
        coins = list(analysis_results['coin_analyses'].keys())
        risk_scores = []
        risk_levels = []
        volume_scores = []
        sentiment_scores = []
        mm_scores = []
        
        for coin_id, analysis in analysis_results['coin_analyses'].items():
            if 'error' not in analysis and 'comprehensive_analysis' in analysis:
                comp = analysis['comprehensive_analysis']
                risk_scores.append(comp['comprehensive_risk_score'])
                risk_levels.append(comp['risk_level'])
                
                # Component scores
                if 'volume_analysis' in analysis and 'error' not in analysis['volume_analysis']:
                    volume_scores.append(analysis['volume_analysis']['indicators']['pump_probability'])
                else:
                    volume_scores.append(0)
                
                if 'sentiment_analysis' in analysis and 'error' not in analysis['sentiment_analysis']:
                    sentiment_scores.append(abs(analysis['sentiment_analysis']['overall_sentiment']) * 50 + 25)
                else:
                    sentiment_scores.append(0)
                
                if 'market_maker_analysis' in analysis and 'error' not in analysis['market_maker_analysis']:
                    mm_scores.append(analysis['market_maker_analysis']['overall_risk_score'])
                else:
                    mm_scores.append(0)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Risk Score by Coin',
                'Component Score Comparison',
                'Risk Level Distribution',
                'Analysis Confidence'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Risk scores bar chart
        fig.add_trace(
            go.Bar(x=coins, y=risk_scores, name="Risk Score", marker_color='red'),
            row=1, col=1
        )
        
        # Component scores comparison
        fig.add_trace(
            go.Bar(x=coins, y=volume_scores, name="Volume Score"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=coins, y=sentiment_scores, name="Sentiment Score"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=coins, y=mm_scores, name="Market Maker Score"),
            row=1, col=2
        )
        
        # Risk level distribution pie chart
        risk_dist = analysis_results['summary']['risk_distribution']
        fig.add_trace(
            go.Pie(
                labels=list(risk_dist.keys()),
                values=list(risk_dist.values()),
                name="Risk Distribution"
            ),
            row=2, col=1
        )
        
        # Analysis confidence
        confidences = []
        for coin_id, analysis in analysis_results['coin_analyses'].items():
            if 'error' not in analysis and 'comprehensive_analysis' in analysis:
                confidences.append(analysis['comprehensive_analysis']['confidence'])
        
        fig.add_trace(
            go.Bar(x=coins, y=confidences, name="Confidence", marker_color='green'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Comprehensive Crypto Pump Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html('/workspace/comprehensive_analysis_dashboard.html')
        fig.show()
    
    def export_analysis_report(self, analysis_results: Dict, filename: str = None) -> str:
        """Export analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/workspace/crypto_pump_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results exported to {filename}")
        return filename


def main():
    """Main function to run comprehensive analysis"""
    analyzer = ComprehensivePumpAnalyzer()
    
    # Define coins to analyze
    coins_to_analyze = [
        {
            'coin_id': 'bitcoin',
            'coin_symbol': 'BTC',
            'coin_address': '0x1234567890123456789012345678901234567890'  # Placeholder
        },
        {
            'coin_id': 'ethereum',
            'coin_symbol': 'ETH',
            'coin_address': '0x1234567890123456789012345678901234567891'
        },
        {
            'coin_id': 'binancecoin',
            'coin_symbol': 'BNB',
            'coin_address': '0x1234567890123456789012345678901234567892'
        },
        {
            'coin_id': 'cardano',
            'coin_symbol': 'ADA',
            'coin_address': '0x1234567890123456789012345678901234567893'
        },
        {
            'coin_id': 'solana',
            'coin_symbol': 'SOL',
            'coin_address': '0x1234567890123456789012345678901234567894'
        }
    ]
    
    print("🚀 Starting Comprehensive Crypto Pump Analysis...")
    print("📊 Analyzing volume patterns, news sentiment, and market maker activity...")
    print("🏦 Focusing on DWF Labs and other major market makers...")
    
    # Run analysis
    results = analyzer.analyze_multiple_coins(coins_to_analyze, days=7)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total coins analyzed: {results['summary']['total_coins_analyzed']}")
    print(f"Successful analyses: {results['summary']['successful_analyses']}")
    print(f"Failed analyses: {results['summary']['failed_analyses']}")
    
    print(f"\nRisk Distribution:")
    for risk_level, count in results['summary']['risk_distribution'].items():
        print(f"  {risk_level}: {count}")
    
    print(f"\nHighest Risk Coins:")
    for i, coin in enumerate(results['summary']['highest_risk_coins'][:5], 1):
        print(f"  {i}. {coin['coin_id']}: {coin['risk_score']:.1f} ({coin['risk_level']})")
    
    # Create dashboard
    print(f"\n📊 Creating comprehensive dashboard...")
    analyzer.create_comprehensive_dashboard(results)
    
    # Export results
    print(f"💾 Exporting analysis results...")
    filename = analyzer.export_analysis_report(results)
    print(f"Results saved to: {filename}")
    
    print(f"\n✅ Analysis complete!")
    print(f"📈 Check the generated HTML files for detailed visualizations")


if __name__ == "__main__":
    main()