# Deep Learning for Portfolio Optimization with Dimensionality Reduction

## Overview

This project aims to implement and extend a deep learning-based approach for portfolio optimization. Drawing inspiration from the paper "Deep Learning for Portfolio Optimization" by Zhang et al., this project integrates dimensionality reduction techniques to enhance the performance and robustness of deep learning models in dynamic financial Tickers. The framework circumvents the need for forecasting expected returns by directly optimizing portfolio weights, focusing on Exchange-Traded Funds (ETFs) across various asset classes.

## Project Goals

1.  **Implement a Deep Learning Model for Portfolio Optimization**:
    *   Develop a neural network model (e.g., LSTM, CNN, or FCN) that takes historical Ticker data as input and outputs optimized portfolio weights.
    *   Optimize the model to directly maximize the Sharpe ratio of the portfolio.

2.  **Integrate Dimensionality Reduction Techniques**:
    *   Incorporate Principal Component Analysis (PCA) and Autoencoders to reduce the dimensionality of input features.
    *   Evaluate the impact of these techniques on model performance, generalization, and computational efficiency.

3.  **Backtest and Evaluate Performance**:
    *   Backtest the model on historical data to assess its risk-adjusted returns and stability.
    *   Compare the performance of the deep learning model with traditional portfolio optimization strategies.

## Key Features

*   **Direct Sharpe Ratio Optimization**: Directly optimize portfolio weights, bypassing return forecasting.
*   **Dimensionality Reduction**: Enhance model performance and reduce overfitting using PCA, t-SNE, and Autoencoders.
*   **Diverse Asset Classes**: Trade Exchange-Traded Funds (ETFs) across stocks, bonds, commodities, and volatility indices.
*   **Dynamic Portfolio Weights**: Adapt portfolio allocations based on Ticker conditions.
*   **Backtesting Framework**: Evaluate performance on historical data using key metrics like Sharpe ratio, Sortino ratio, and maximum drawdown.

## Technologies Used

*   **Programming Language**: Python
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: TensorFlow, Keras, Scikit-learn
*   **Data Visualization**: Matplotlib, Seaborn
*   **Data Sources**:
    *   Yahoo Finance (yfinance)
    *   FRED Api


## Project Structure
project_folder/
│
├── data_scraper.py          # Scrapes raw ETF data
├── preprocessor.py          # Cleans and preprocesses raw data
├── feature_engineering.py   # Generates advanced features
├── dimensionality_reduction.py # Applies PCA/t-SNE/Autoencoders
├── model_trainer.py         # Trains deep learning models
├── portfolio_optimizer.py   # Optimizes portfolio weights
├── backtesting.py           # Evaluates portfolio performance
├── visualization_dashboard.py # Creates visualizations
│
├── data/                    # Raw scraped data
├── processed_data/          # Preprocessed datasets
├── features/                # Feature matrices
├── reduced_data/            # Dimensionality-reduced datasets
├── models/                  # Trained deep learning models
├── portfolio_weights/       # Optimized portfolio weights
└── results/                 # Backtesting results and metrics

### Modules

*   **`data_scraper.py`**: Scrapes historical ETF and cryptocurrency data from Yahoo Finance, Alpha Vantage, and CoinTickerCap.
*   **`preprocessor.py`**: Cleans and preprocesses raw data, handling missing values and scaling.
*   **`feature_engineering.py`**: Generates technical indicators and rolling correlations from preprocessed data.
*   **`dimensionality_reduction.py`**: Applies PCA and Autoencoders to reduce the dimensionality of the feature space.
*   **`model_trainer.py`**: Trains deep learning models (e.g., LSTM, CNN) to predict portfolio weights.
*   **`portfolio_optimizer.py`**: Uses trained models to generate optimized portfolio weights.
*   **`backtesting.py`**: Backtests the portfolio strategy on historical data and calculates performance metrics.
*   **`visualization_dashboard.py`**: Creates interactive visualizations of data, model performance, and portfolio allocations.

