

## Workflow

### 1. Data Acquisition

* Collected one month of historical data for **Apple (AAPL)** and the **S&P 500 (SPY)** using `yfinance`.

### 2. Data Storage

* Stored raw price and volume data in **MongoDB**.
* Transformed and stored model-ready returns and engineered features in **SQLite**.



### 3. Exploratory Data Analysis
* Analyzed price trends, daily returns, and volatility.
* Examined relationships between market variables using correlation analysis.
* Used the Shapiro-Wilk test to assess the normality of stock returns.
* Calculated a 5-day Simple Moving Average (SMA) to examine short-term price trends.

### 4. Feature Engineering

* Created lagged-return features.
* Calculated 5-day volatility.
* Created an up/down classification target for predicting stock-price direction.

### 5. Machine Learning

* Used **Linear Regression** and **Random Forest Regression** to predict returns.
* Built a **Random Forest Classifier** to predict price direction and generate trading signals.

### 6. Strategy Backtesting

* Developed a model-based trading strategy.
* Compared the strategy with **Buy & Hold** using cumulative returns.
* Evaluated performance using the **Sharpe Ratio** and directional accuracy.

### 7. Time-Series Forecasting

* Applied an **ARIMA(5,1,0)** model for short-term Apple stock-price forecasting.

## Project Value

This project demonstrates an end-to-end financial analytics workflow combining:

* Financial data acquisition and processing
* MongoDB and SQLite data storage
* Exploratory data analysis and feature engineering
* Machine learning
* Time-series forecasting
* Trading-strategy backtesting and performance evaluation


## Repository Structure

```text
aapl-strategy-backtesting/
├── data/
│   └── stock_comparison_df.csv
├── notebooks/
│   └── aapl_strategy_backtesting.ipynb
└── README.md
```
