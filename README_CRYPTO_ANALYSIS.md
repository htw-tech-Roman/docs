# Crypto Pump Analysis System

Comprehensive analysis system for detecting cryptocurrency pump activities, considering volume anomalies, news sentiment, and market maker activities (with focus on DWF Labs).

## Features

### 📊 Volume Analysis
- Real-time data from CoinGecko API
- Volume anomaly detection using statistical methods
- Price action analysis and volatility metrics
- Pump probability scoring

### 📰 News Sentiment Analysis
- Multi-source sentiment analysis (Reddit, news, social media)
- Pump/dump keyword detection
- Manipulation risk assessment
- Sentiment trend analysis

### 🏦 Market Maker Analysis
- DWF Labs activity monitoring
- Other major market makers (Alameda, Jump Trading, etc.)
- Suspicious trading pattern detection
- Risk scoring and alerts

### 🎯 Comprehensive Risk Assessment
- Weighted scoring system
- Risk level classification (CRITICAL, HIGH, MODERATE, LOW, MINIMAL)
- Actionable recommendations
- Real-time monitoring capabilities

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python comprehensive_pump_analyzer.py
```

3. Or use the Jupyter notebook for interactive analysis:
```bash
jupyter notebook crypto_pump_analysis.ipynb
```

## Usage

### Basic Analysis
```python
from comprehensive_pump_analyzer import ComprehensivePumpAnalyzer

analyzer = ComprehensivePumpAnalyzer()

# Analyze a single coin
result = analyzer.analyze_coin(
    coin_id='bitcoin',
    coin_symbol='BTC',
    coin_address='0x1234...',  # Optional
    days=7
)

# Analyze multiple coins
coins = [
    {'coin_id': 'bitcoin', 'coin_symbol': 'BTC'},
    {'coin_id': 'ethereum', 'coin_symbol': 'ETH'},
    # ... more coins
]

results = analyzer.analyze_multiple_coins(coins, days=7)
```

### Real-time Monitoring
```python
# Monitor a specific coin
analyzer.monitor_coin('bitcoin', 'BTC', interval_minutes=30)
```

## Analysis Components

### 1. Volume Analyzer (`crypto_pump_analyzer.py`)
- Fetches data from CoinGecko API
- Detects volume anomalies using rolling statistics
- Calculates pump probability scores
- Analyzes price action and volatility

### 2. News Sentiment Analyzer (`news_sentiment_analyzer.py`)
- Analyzes sentiment from multiple sources
- Detects pump/dump keywords
- Calculates manipulation risk
- Provides sentiment trends

### 3. Market Maker Analyzer (`market_maker_analyzer.py`)
- Monitors DWF Labs and other market makers
- Detects suspicious trading patterns
- Analyzes wash trading and manipulation
- Provides risk assessments

### 4. Comprehensive Analyzer (`comprehensive_pump_analyzer.py`)
- Integrates all analysis components
- Calculates weighted risk scores
- Generates final recommendations
- Creates interactive dashboards

## Risk Levels

- **CRITICAL (80-100)**: Extreme risk, immediate action required
- **HIGH (60-79)**: High risk, reduce positions and monitor closely
- **MODERATE (40-59)**: Moderate risk, stay alert
- **LOW (20-39)**: Low risk, normal monitoring
- **MINIMAL (0-19)**: Minimal risk, normal market conditions

## Market Makers Monitored

- **DWF Labs**: Primary focus, known for coordinated activities
- **Alameda Research**: High-risk, wash trading patterns
- **Jump Trading**: Algorithmic trading patterns
- **Wintermute**: Market making activities
- **GSR**: Institutional trading

## Generated Outputs

1. **Interactive Dashboards**: HTML files with Plotly visualizations
2. **Analysis Reports**: JSON files with detailed results
3. **Summary Reports**: High-level overview of findings
4. **Real-time Alerts**: Console output for monitoring

## Configuration

### Risk Thresholds
```python
volume_thresholds = {
    'extreme': 10.0,  # 10x normal volume
    'high': 5.0,      # 5x normal volume
    'moderate': 2.0   # 2x normal volume
}
```

### Analysis Weights
```python
weights = {
    'volume_analysis': 0.4,      # 40% weight
    'sentiment_analysis': 0.3,   # 30% weight
    'market_maker_analysis': 0.3  # 30% weight
}
```

## API Keys Required

For full functionality, you'll need API keys for:
- CoinGecko API (free tier available)
- News APIs (NewsAPI, CryptoPanic)
- Social media APIs (Twitter, Reddit)
- Blockchain explorers (Etherscan, etc.)

## Example Analysis Results

```
ANALYSIS SUMMARY
================================================================================
Total coins analyzed: 8
Successful analyses: 8
Failed analyses: 0

Risk Distribution:
  CRITICAL: 1
  HIGH: 2
  MODERATE: 3
  LOW: 2
  MINIMAL: 0

Highest Risk Coins:
  1. solana: 85.2 (CRITICAL)
  2. cardano: 72.1 (HIGH)
  3. chainlink: 68.5 (HIGH)
```

## Recommendations

The system provides specific recommendations based on analysis results:

- **Risk Management**: Position sizing, stop-loss orders
- **Monitoring**: Real-time alerts, pattern detection
- **Trading**: Entry/exit strategies, volatility management

## Disclaimer

This tool is for educational and research purposes only. It does not provide financial advice. Always do your own research and consider your risk tolerance before making investment decisions.

## Contributing

Feel free to contribute improvements, additional data sources, or new analysis methods.

## License

MIT License - see LICENSE file for details.