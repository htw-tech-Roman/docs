#!/usr/bin/env python3
"""
News Sentiment Analyzer for Crypto Pump Detection
Integrates multiple news sources and sentiment analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for crypto assets"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # News sources configuration
        self.news_sources = {
            'cryptopanic': {
                'base_url': 'https://cryptopanic.com/api/v1/posts/',
                'api_key': None,  # Would need API key
                'enabled': False
            },
            'newsapi': {
                'base_url': 'https://newsapi.org/v2/everything',
                'api_key': None,  # Would need API key
                'enabled': False
            },
            'reddit': {
                'base_url': 'https://www.reddit.com/r/cryptocurrency/hot.json',
                'enabled': True
            },
            'twitter': {
                'enabled': False  # Would need Twitter API
            }
        }
        
        # Keywords for pump/dump detection
        self.pump_keywords = [
            'pump', 'moon', 'rocket', 'bullish', 'breakout', 'surge', 'rally',
            'explosive', 'massive', 'huge', 'gains', 'profit', 'buy', 'hodl',
            'diamond hands', 'to the moon', 'lambo', 'wen moon'
        ]
        
        self.dump_keywords = [
            'dump', 'crash', 'bearish', 'sell', 'panic', 'fear', 'red',
            'loss', 'down', 'fall', 'drop', 'correction', 'bear market',
            'paper hands', 'sell off', 'bloodbath'
        ]
        
        self.manipulation_keywords = [
            'manipulation', 'pump and dump', 'coordinated', 'whale', 'insider',
            'market maker', 'wash trading', 'bot', 'algorithm', 'coordinated attack'
        ]
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using multiple methods"""
        if not text or len(text.strip()) == 0:
            return {'sentiment': 0, 'confidence': 0, 'method': 'none'}
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        vader_sentiment = vader_scores['compound']
        
        # Keyword-based sentiment
        keyword_sentiment = self._analyze_keyword_sentiment(text)
        
        # Combine results
        sentiments = [textblob_sentiment, vader_sentiment, keyword_sentiment]
        valid_sentiments = [s for s in sentiments if s is not None]
        
        if not valid_sentiments:
            return {'sentiment': 0, 'confidence': 0, 'method': 'none'}
        
        avg_sentiment = np.mean(valid_sentiments)
        confidence = 1.0 - np.std(valid_sentiments) if len(valid_sentiments) > 1 else 1.0
        
        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'textblob': textblob_sentiment,
            'vader': vader_sentiment,
            'keyword': keyword_sentiment,
            'method': 'combined'
        }
    
    def _analyze_keyword_sentiment(self, text: str) -> float:
        """Analyze sentiment based on pump/dump keywords"""
        text_lower = text.lower()
        
        pump_count = sum(1 for keyword in self.pump_keywords if keyword in text_lower)
        dump_count = sum(1 for keyword in self.dump_keywords if keyword in text_lower)
        manipulation_count = sum(1 for keyword in self.manipulation_keywords if keyword in text_lower)
        
        total_keywords = pump_count + dump_count + manipulation_count
        
        if total_keywords == 0:
            return 0.0
        
        # Calculate sentiment based on keyword ratios
        pump_ratio = pump_count / total_keywords
        dump_ratio = dump_count / total_keywords
        manipulation_ratio = manipulation_count / total_keywords
        
        # Manipulation keywords reduce confidence
        manipulation_penalty = manipulation_ratio * 0.5
        
        sentiment = (pump_ratio - dump_ratio) * (1 - manipulation_penalty)
        
        return max(-1.0, min(1.0, sentiment))
    
    def get_reddit_sentiment(self, coin_symbol: str, limit: int = 100) -> Dict:
        """Get sentiment from Reddit posts"""
        try:
            # This is a simplified version - in practice, you'd use Reddit API
            # For now, we'll simulate Reddit data
            reddit_posts = self._simulate_reddit_data(coin_symbol, limit)
            
            sentiments = []
            pump_mentions = 0
            dump_mentions = 0
            
            for post in reddit_posts:
                sentiment = self.analyze_text_sentiment(post['text'])
                sentiments.append(sentiment['sentiment'])
                
                if any(keyword in post['text'].lower() for keyword in self.pump_keywords):
                    pump_mentions += 1
                if any(keyword in post['text'].lower() for keyword in self.dump_keywords):
                    dump_mentions += 1
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
            
            return {
                'source': 'reddit',
                'avg_sentiment': avg_sentiment,
                'sentiment_std': sentiment_std,
                'post_count': len(reddit_posts),
                'pump_mentions': pump_mentions,
                'dump_mentions': dump_mentions,
                'pump_ratio': pump_mentions / len(reddit_posts) if reddit_posts else 0,
                'confidence': 1.0 - sentiment_std if sentiment_std < 1.0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {str(e)}")
            return {'source': 'reddit', 'error': str(e)}
    
    def _simulate_reddit_data(self, coin_symbol: str, limit: int) -> List[Dict]:
        """Simulate Reddit data for demonstration"""
        # In practice, this would fetch real Reddit data
        sample_posts = [
            f"${coin_symbol} is going to the moon! 🚀",
            f"Just bought more {coin_symbol}, diamond hands!",
            f"{coin_symbol} pump incoming, get ready!",
            f"${coin_symbol} looking bullish today",
            f"Anyone else holding {coin_symbol}?",
            f"{coin_symbol} price action is crazy",
            f"${coin_symbol} breakout confirmed!",
            f"Thinking about selling {coin_symbol}",
            f"{coin_symbol} correction incoming",
            f"${coin_symbol} whale activity detected"
        ]
        
        posts = []
        for i in range(min(limit, len(sample_posts))):
            posts.append({
                'text': sample_posts[i % len(sample_posts)],
                'timestamp': datetime.now() - timedelta(hours=i),
                'score': np.random.randint(1, 100),
                'comments': np.random.randint(0, 50)
            })
        
        return posts
    
    def get_crypto_news_sentiment(self, coin_symbol: str, hours: int = 24) -> Dict:
        """Get sentiment from crypto news sources"""
        try:
            # Simulate news data - in practice, integrate with real news APIs
            news_articles = self._simulate_news_data(coin_symbol, hours)
            
            sentiments = []
            pump_articles = 0
            dump_articles = 0
            manipulation_articles = 0
            
            for article in news_articles:
                sentiment = self.analyze_text_sentiment(article['title'] + ' ' + article['content'])
                sentiments.append(sentiment['sentiment'])
                
                text = (article['title'] + ' ' + article['content']).lower()
                if any(keyword in text for keyword in self.pump_keywords):
                    pump_articles += 1
                if any(keyword in text for keyword in self.dump_keywords):
                    dump_articles += 1
                if any(keyword in text for keyword in self.manipulation_keywords):
                    manipulation_articles += 1
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
            
            return {
                'source': 'crypto_news',
                'avg_sentiment': avg_sentiment,
                'sentiment_std': sentiment_std,
                'article_count': len(news_articles),
                'pump_articles': pump_articles,
                'dump_articles': dump_articles,
                'manipulation_articles': manipulation_articles,
                'pump_ratio': pump_articles / len(news_articles) if news_articles else 0,
                'manipulation_ratio': manipulation_articles / len(news_articles) if news_articles else 0,
                'confidence': 1.0 - sentiment_std if sentiment_std < 1.0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting crypto news sentiment: {str(e)}")
            return {'source': 'crypto_news', 'error': str(e)}
    
    def _simulate_news_data(self, coin_symbol: str, hours: int) -> List[Dict]:
        """Simulate news data for demonstration"""
        sample_articles = [
            {
                'title': f'{coin_symbol} Surges 20% on Major Partnership Announcement',
                'content': f'Cryptocurrency {coin_symbol} experienced a significant price surge following the announcement of a major partnership with a leading technology company. The news has sparked renewed investor interest and positive market sentiment.',
                'source': 'CryptoNews',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'title': f'Analysts Predict {coin_symbol} Breakout as Volume Spikes',
                'content': f'Technical analysts are pointing to unusual volume patterns in {coin_symbol} trading, suggesting a potential breakout. The cryptocurrency has been consolidating for weeks and may be ready for a significant move.',
                'source': 'CoinDesk',
                'timestamp': datetime.now() - timedelta(hours=5)
            },
            {
                'title': f'{coin_symbol} Faces Selling Pressure as Whales Move Coins',
                'content': f'Large holders of {coin_symbol} have been observed moving significant amounts of the cryptocurrency to exchanges, raising concerns about potential selling pressure. Market participants are watching closely for signs of coordinated selling.',
                'source': 'CryptoSlate',
                'timestamp': datetime.now() - timedelta(hours=8)
            },
            {
                'title': f'Regulatory Concerns Weigh on {coin_symbol} Market',
                'content': f'Recent regulatory developments have created uncertainty in the {coin_symbol} market. Investors are cautious as authorities consider new regulations that could impact the cryptocurrency sector.',
                'source': 'TheBlock',
                'timestamp': datetime.now() - timedelta(hours=12)
            },
            {
                'title': f'{coin_symbol} Community Celebrates Major Milestone',
                'content': f'The {coin_symbol} community is celebrating a major milestone as the cryptocurrency reaches new adoption levels. The positive sentiment is reflected in increased social media activity and community engagement.',
                'source': 'Decrypt',
                'timestamp': datetime.now() - timedelta(hours=18)
            }
        ]
        
        # Filter articles within the specified time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_articles = [article for article in sample_articles if article['timestamp'] > cutoff_time]
        
        return filtered_articles
    
    def get_social_media_sentiment(self, coin_symbol: str) -> Dict:
        """Get sentiment from social media platforms"""
        # This would integrate with Twitter API, Telegram, Discord, etc.
        # For now, we'll simulate social media sentiment
        
        social_data = {
            'twitter': {
                'mentions': np.random.randint(100, 1000),
                'sentiment': np.random.uniform(-0.5, 0.5),
                'pump_mentions': np.random.randint(0, 50),
                'dump_mentions': np.random.randint(0, 30)
            },
            'telegram': {
                'mentions': np.random.randint(50, 500),
                'sentiment': np.random.uniform(-0.3, 0.7),
                'pump_mentions': np.random.randint(0, 30),
                'dump_mentions': np.random.randint(0, 20)
            },
            'discord': {
                'mentions': np.random.randint(20, 200),
                'sentiment': np.random.uniform(-0.2, 0.6),
                'pump_mentions': np.random.randint(0, 15),
                'dump_mentions': np.random.randint(0, 10)
            }
        }
        
        # Calculate overall social sentiment
        all_sentiments = [data['sentiment'] for data in social_data.values()]
        avg_sentiment = np.mean(all_sentiments)
        
        total_pump_mentions = sum(data['pump_mentions'] for data in social_data.values())
        total_dump_mentions = sum(data['dump_mentions'] for data in social_data.values())
        total_mentions = sum(data['mentions'] for data in social_data.values())
        
        return {
            'source': 'social_media',
            'avg_sentiment': avg_sentiment,
            'total_mentions': total_mentions,
            'pump_mentions': total_pump_mentions,
            'dump_mentions': total_dump_mentions,
            'pump_ratio': total_pump_mentions / total_mentions if total_mentions > 0 else 0,
            'platforms': social_data,
            'confidence': 0.7  # Social media sentiment is generally less reliable
        }
    
    def get_comprehensive_sentiment(self, coin_symbol: str) -> Dict:
        """Get comprehensive sentiment analysis from all sources"""
        logger.info(f"Analyzing sentiment for {coin_symbol}")
        
        # Get sentiment from different sources
        reddit_sentiment = self.get_reddit_sentiment(coin_symbol)
        news_sentiment = self.get_crypto_news_sentiment(coin_symbol)
        social_sentiment = self.get_social_media_sentiment(coin_symbol)
        
        # Combine all sentiment data
        all_sources = [reddit_sentiment, news_sentiment, social_sentiment]
        valid_sources = [s for s in all_sources if 'error' not in s]
        
        if not valid_sources:
            return {'error': 'No valid sentiment sources available'}
        
        # Calculate weighted average sentiment
        sentiments = []
        confidences = []
        pump_ratios = []
        
        for source in valid_sources:
            if 'avg_sentiment' in source:
                sentiments.append(source['avg_sentiment'])
                confidences.append(source.get('confidence', 0.5))
                pump_ratios.append(source.get('pump_ratio', 0))
        
        # Weight by confidence
        if confidences:
            weights = np.array(confidences) / np.sum(confidences)
            weighted_sentiment = np.average(sentiments, weights=weights)
        else:
            weighted_sentiment = np.mean(sentiments) if sentiments else 0
        
        # Calculate overall metrics
        total_pump_mentions = sum(source.get('pump_mentions', 0) for source in valid_sources)
        total_dump_mentions = sum(source.get('dump_mentions', 0) for source in valid_sources)
        total_mentions = sum(source.get('mentions', source.get('post_count', source.get('article_count', 0))) for source in valid_sources)
        
        # Determine sentiment trend
        if weighted_sentiment > 0.3:
            trend = 'bullish'
        elif weighted_sentiment < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Calculate manipulation risk
        manipulation_risk = 0
        for source in valid_sources:
            if 'manipulation_ratio' in source:
                manipulation_risk += source['manipulation_ratio']
        
        manipulation_risk = min(manipulation_risk, 1.0)
        
        return {
            'coin_symbol': coin_symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': weighted_sentiment,
            'sentiment_trend': trend,
            'confidence': np.mean(confidences) if confidences else 0.5,
            'total_mentions': total_mentions,
            'pump_mentions': total_pump_mentions,
            'dump_mentions': total_dump_mentions,
            'pump_ratio': total_pump_mentions / total_mentions if total_mentions > 0 else 0,
            'manipulation_risk': manipulation_risk,
            'sources': {
                'reddit': reddit_sentiment,
                'news': news_sentiment,
                'social_media': social_sentiment
            },
            'recommendations': self._generate_sentiment_recommendations(weighted_sentiment, trend, manipulation_risk)
        }
    
    def _generate_sentiment_recommendations(self, sentiment: float, trend: str, manipulation_risk: float) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []
        
        if trend == 'bullish' and sentiment > 0.5:
            recommendations.append("📈 Strong positive sentiment detected - may support upward price movement")
        elif trend == 'bearish' and sentiment < -0.5:
            recommendations.append("📉 Strong negative sentiment detected - may pressure price downward")
        elif trend == 'neutral':
            recommendations.append("😐 Neutral sentiment - market appears balanced")
        
        if manipulation_risk > 0.3:
            recommendations.append("⚠️ High manipulation risk detected in social sentiment")
        elif manipulation_risk > 0.1:
            recommendations.append("⚡ Moderate manipulation risk - monitor for coordinated activity")
        
        if sentiment > 0.7:
            recommendations.append("🚀 Extremely bullish sentiment - be cautious of potential FOMO")
        elif sentiment < -0.7:
            recommendations.append("😱 Extremely bearish sentiment - potential buying opportunity")
        
        return recommendations


def main():
    """Test the sentiment analyzer"""
    analyzer = NewsSentimentAnalyzer()
    
    # Test with a sample coin
    coin_symbol = "BTC"
    sentiment_data = analyzer.get_comprehensive_sentiment(coin_symbol)
    
    print(f"Sentiment Analysis for {coin_symbol}:")
    print(f"Overall Sentiment: {sentiment_data['overall_sentiment']:.3f}")
    print(f"Trend: {sentiment_data['sentiment_trend']}")
    print(f"Confidence: {sentiment_data['confidence']:.3f}")
    print(f"Pump Ratio: {sentiment_data['pump_ratio']:.3f}")
    print(f"Manipulation Risk: {sentiment_data['manipulation_risk']:.3f}")
    
    print("\nRecommendations:")
    for rec in sentiment_data['recommendations']:
        print(f"  {rec}")


if __name__ == "__main__":
    main()