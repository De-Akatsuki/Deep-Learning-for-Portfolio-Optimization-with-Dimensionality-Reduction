
# Deep Learning Portfolio Optimization with Dimensionality Reduction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Overview

This project implements a comprehensive deep learning-based portfolio optimization framework that leverages dimensionality reduction techniques to enhance performance and robustness in dynamic financial markets. The system focuses on Exchange-Traded Funds (ETFs) across various asset classes and circumvents traditional return forecasting by directly optimizing portfolio weights to maximize risk-adjusted returns.

### 🎯 Key Innovation

The framework integrates multiple dimensionality reduction techniques (PCA, Autoencoders, Hybrid approaches) with state-of-the-art deep learning architectures to create an end-to-end portfolio optimization pipeline that adapts to changing market conditions.

## 🚀 Features

### Core Capabilities

- **Direct Sharpe Ratio Optimization**: Bypasses return forecasting by directly optimizing portfolio weights
- **Multi-Model Architecture**: Implements CNN, LSTM, Bi-LSTM, CNN-LSTM, TCN, and Transformer models
- **Advanced Dimensionality Reduction**: PCA, Autoencoders, and Hybrid approaches
- **Comprehensive Feature Engineering**: 100+ technical indicators and macroeconomic variables
- **Real-time Data Integration**: Yahoo Finance and FRED API integration
- **Robust Backtesting Framework**: Performance evaluation with multiple risk metrics

### Supported Models

- **Baseline Models**: Multi-Layer Perceptron (MLP)
- **Sequential Models**: LSTM, Bi-LSTM, CNN-LSTM
- **Convolutional Models**: CNN, TCN (Temporal Convolutional Networks)
- **Attention-Based Models**: Transformers, Hybrid Transformer-BiLSTM
- **Dimensionality Reduction**: PCA, Autoencoders, Hybrid PCA-Autoencoder

## 📁 Project Structure

```
Dimensionality-Reduction-PortfolioOptimization-backup/
│
├── 📊 EDA/                                    # Exploratory Data Analysis
│   ├── 📈 ETFs_datasets/                      # Processed datasets
│   │   ├── Final_csv/                         # Final processed datasets
│   │   ├── Model Data/                        # Model-specific data
│   │   │   ├── PCA/                          # PCA components
│   │   │   └── Pre-Dim Reduction/            # Pre-dimensionality reduction data
│   │   └── RAW/                              # Raw data sources
│   │       ├── ETFS/                         # ETF historical data
│   │       └── FRED/                         # Macroeconomic data
│   │
│   └── 📓 Notebooks/                         # Jupyter notebooks
│       ├── datacollection_preprocessing/     # Data collection & preprocessing
│       ├── deep_learning_models/            # Model implementations
│       │   ├── DR_models/                   # Models with dimensionality reduction
│       │   │   ├── BI-LSTM/                 # Bidirectional LSTM models
│       │   │   ├── CNN/                     # Convolutional Neural Networks
│       │   │   ├── CNN-LSTM/                # Hybrid CNN-LSTM models
│       │   │   ├── Hybrid_Transformer/      # Transformer + Bi-LSTM
│       │   │   ├── LSTM/                    # LSTM models
│       │   │   ├── TCN/                     # Temporal Convolutional Networks
│       │   │   └── Transformers/            # Transformer models
│       │   └── Without_DR/                  # Baseline models without DR
│       ├── dimensionality_reduction_techniques/ # DR technique implementations
│       └── images/                          # Visualizations and diagrams
│
├── ⚙️ config.py                              # Configuration management
├── 📋 dependencies.txt                       # Python dependencies
├── 🔑 API_KEYS.env                          # API keys configuration
├── 📄 LICENSE                               # MIT License
└── 📖 README.md                             # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Dimensionality-Reduction-PortfolioOptimization.git
   cd Dimensionality-Reduction-PortfolioOptimization
   ```
2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r dependencies.txt
   ```
4. **Set up API keys**

   ```bash
   cp API_KEYS.env.example API_KEYS.env
   # Edit API_KEYS.env with your FRED API key
   ```

### Required API Keys

- **FRED API Key**: Get your free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)

## 🚀 Quick Start

### 1. Data Collection

```python
# Run the ETF data scraper
python EDA/Notebooks/datacollection_preprocessing/1a-data_scraper(ETF).py

# Run the macroeconomic data scraper
python EDA/Notebooks/datacollection_preprocessing/1b-macro_scrapper(FRED).py
```

### 2. Data Preprocessing

```python
# Open and run the preprocessing notebook
jupyter notebook EDA/Notebooks/datacollection_preprocessing/4a-pre-processing(cleaning).ipynb
```

### 3. Feature Engineering

```python
# Generate technical indicators and features
python EDA/Notebooks/datacollection_preprocessing/3-feature_engineering_new_Indicators.py
```

### 4. Dimensionality Reduction

```python
# Apply PCA
jupyter notebook EDA/Notebooks/dimensionality_reduction_techniques/PCA.ipynb

# Apply Autoencoders
jupyter notebook EDA/Notebooks/dimensionality_reduction_techniques/Auto-encoder.ipynb
```

### 5. Model Training

```python
# Train baseline models
jupyter notebook EDA/Notebooks/deep_learning_models/Without_DR/baseline_model.ipynb

# Train models with dimensionality reduction
jupyter notebook EDA/Notebooks/deep_learning_models/DR_models/CNN/CNN (PCA).ipynb
```

## 📊 Dataset Information

### ETF Data

- **Assets**: VTI (Total Stock Market), AGG (Bond Market), DBC (Commodities), VIX (Volatility)
- **Features**: 100+ technical indicators and macroeconomic variables
- **Time Period**: 10 years of historical data
- **Frequency**: Daily data

### Technical Indicators

- **Moving Averages**: SMA, EMA (10, 20, 30, 50, 100, 200 periods)
- **Bollinger Bands**: Upper, Middle, Lower bands
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillator
- **Volatility Indicators**: ATR, ADX, DMI
- **Volume Indicators**: OBV
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio

### Macroeconomic Variables

- **Inflation**: CPI, Core CPI, PCE Price Index
- **Economic Growth**: Real GDP, Industrial Production
- **Employment**: Unemployment Rate, Nonfarm Payrolls
- **Interest Rates**: Fed Funds Rate, Treasury Yields
- **Housing**: Housing Starts, Building Permits
- **Consumer**: Consumer Sentiment, Credit

## 🧠 Model Architectures

### 1. Baseline Models

- **Multi-Layer Perceptron (MLP)**: Feedforward neural network with fully connected layers
- **Naive Equal Weight**: Equal allocation across all assets
- **Black-Litterman**: Traditional portfolio optimization

### 2. Sequential Models

- **LSTM**: Long Short-Term Memory networks for temporal dependencies
- **Bi-LSTM**: Bidirectional LSTM for enhanced pattern recognition
- **CNN-LSTM**: Hybrid architecture combining convolutional and recurrent layers

### 3. Convolutional Models

- **CNN**: Convolutional Neural Networks for feature extraction
- **TCN**: Temporal Convolutional Networks with dilated convolutions

### 4. Attention-Based Models

- **Transformers**: Self-attention mechanisms for long-range dependencies
- **Hybrid Transformer-BiLSTM**: Combining transformer attention with LSTM memory

### 5. Dimensionality Reduction Techniques

- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **Autoencoders**: Neural network-based nonlinear dimensionality reduction
- **Hybrid PCA-Autoencoder**: Combining both approaches for optimal feature representation

## 📈 Performance Metrics

The framework evaluates portfolio performance using multiple risk-adjusted metrics:

- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Volatility**: Standard deviation of returns
- **Beta**: Systematic risk measure
- **Alpha**: Excess return over market
- **Win Rate**: Percentage of profitable periods
- **Up/Down Capture Ratios**: Performance in up/down markets

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
FRED_API_KEY=your_fred_api_key_here
```

### Model Configuration

Key parameters can be adjusted in the respective notebook files:

- **Sequence Length**: Historical window for predictions
- **Hidden Units**: Number of neurons in hidden layers
- **Learning Rate**: Optimization step size
- **Batch Size**: Training batch size
- **Epochs**: Number of training iterations

## 📊 Results and Visualizations

The project includes comprehensive visualizations:

- **Architecture Diagrams**: Model structure visualizations
- **Performance Metrics**: Risk-return analysis
- **Portfolio Weights**: Dynamic allocation over time
- **Correlation Matrices**: Asset and feature relationships
- **Time Series Plots**: Historical performance tracking

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📚 Research Background

This project is inspired by research in:

- **Deep Learning for Portfolio Optimization** (Zhang et al.)
- **Dimensionality Reduction in Financial Time Series**
- **Risk-Adjusted Portfolio Optimization**
- **Machine Learning in Quantitative Finance**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Corchil Kelly Kwame** - *Initial work* - [https://github.com/De-Akatsuki](https://github.com/De-Akatsuki)

## 🙏 Acknowledgments

- University of Ghana Data Science Lab
- FRED API for macroeconomic data
- Yahoo Finance for ETF data
- Open source machine learning libraries (TensorFlow, PyTorch, scikit-learn)

## 📞 Contact

For questions or collaboration opportunities:

- Email: [[von_stark@outlook.com](von_stark@outlook.com)]
- LinkedIn: [[https://www.linkedin.com/in/corchil-kelly/](https://www.linkedin.com/in/corchil-kelly/)]

## 🔮 Future Work

- [ ] Real-time trading integration with broker platforms
- [ ] Additional asset classes (cryptocurrencies, forex)
- [ ] Reinforcement learning approaches
- [ ] Multi-objective optimization
- [ ] Cloud deployment and API development
- [ ] Interactive web dashboard

---

**⚠️ Disclaimer**: This project is for educational and research purposes only. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.
