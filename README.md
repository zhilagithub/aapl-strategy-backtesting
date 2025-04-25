

🔧 Workflow Breakdown:
Data Acquisition

Collected 1-month historical data for Apple (AAPL) and S&P 500 (SPY) using yfinance.
Data Storage

Stored raw price and volume data in MongoDB (ideal for semi-structured financial feeds).
Transformed and saved model-ready returns and engineered features in SQLite (optimized for relational analytics).
Exploratory Data Analysis (EDA)

Visualized price trends, daily returns, and volatility analysis.
Created a correlation heatmap for technical insights.
5-day moving average (SMA) to analyze short-term trends and smooth price fluctuations.
Feature Engineering

Engineered features such as lagged returns, 5-day volatility, and directional (up/down) target classification for model input.
Machine Learning Models

Applied Linear Regression and Random Forest Regressor for predicting returns.
Developed Random Forest Classifier to generate trading signals based on price movements.
Strategy Backtesting

Created a model-driven trading strategy.
Compared it with the Buy & Hold strategy using cumulative returns.
Evaluated both strategies with performance metrics such as Sharpe Ratio and directional accuracy.
Forecasting with ARIMA

Applied the ARIMA(5,1,0) model to forecast short-term stock prices for Apple.
🧠 Real-World Value
This notebook demonstrates a complete quantitative research and trading strategy pipeline, showcasing:

Hybrid storage systems (MongoDB + SQLite).
Backtesting and forecasting capabilities.
End-to-end reproducibility, from data collection to actionable insights.
