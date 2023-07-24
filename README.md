# LSTM-RF-Stocks-Forecasting-Dashboard

## Overview

This repository contains a machine learning-based stock prediction and analysis project that utilizes Random Forest and LSTM models to predict stock movements and future prices, respectively. The project also includes a dynamic dashboard that provides valuable insights into stock data, forecasted trends using Facebook Prophet, and displays accounting data and relevant news about the stock.

**Stock Prediction:**

- The Random Forest model leverages historical stock market data, technical indicators, and other relevant features to predict whether a stock's price will increase, decrease, or remain stable at the next time step. The model utilizes the power of ensemble learning to capture complex relationships and patterns in the data.

**Future Price Forecasting:**

- The LSTM (Long Short-Term Memory) model is employed for predicting future stock prices. LSTM is a type of recurrent neural network that excels at capturing temporal dependencies in time-series data. It considers the historical price trends and other relevant factors to forecast future stock prices.

**Dynamic Dashboard:**

- The dashboard provides a user-friendly interface to interact with the stock prediction and analysis results. Users can easily visualize historical stock data, predictions, and forecasts through various interactive charts and graphs.
- The dashboard is built using Streamlit, a Python web framework for building analytical web applications. It allows users to select specific stocks, time periods, and view various analysis insights.

**Facebook Prophet Forecasting:**

- As an additional forecasting method, the dashboard incorporates Facebook Prophet, a time-series forecasting library developed by Facebook's Core Data Science team. It can provide valuable insights into stock price trends, seasonal patterns, and potential future movements.

**Accounting Data and News:**

- The dashboard also displays relevant accounting data for the selected stock, such as earnings per share, price-to-earnings ratio, and other financial metrics. This information can help investors make more informed decisions.
- Additionally, the dashboard integrates a news section that fetches and displays the latest news related to the selected stock. Staying updated with the latest news can provide valuable context to stock market movements.

## Features

- Predicts stock movements using Random Forest.
- Forecasts future stock prices using LSTM.
- Provides real-time stock data visualization with a dynamic dashboard.
- Utilizes Facebook Prophet to forecast trends in stock prices.
- Displays accounting data and relevant news about the stock.

## Installation

The following packages are required to run the code in this repository:

- streamlit, pandas, sklearn, tensorflow, numpy, yahoo finance, dotenv, stocknews, datetime, prophet, plotly

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for training and evaluating the machine learning models consists of historical stock market data of the S&P 500 Index. It includes features such as past stock prices, trading volume, technical indicators, and any other relevant factors that can potentially influence stock movements and prices. The dataset is preprocessed to handle missing values, normalize data, and prepare it in a suitable format for training the Random Forest and LSTM models. The dataset used for the dashboard is fetched live on user request from Yahoo Finance.

## Notebook

There are two Jupyter notebooks included, `stocks_models.ipynb` and `dashboard_workbook.ipynb`. The former highlights my process of building and training the Random Forest and LSTM models with detailed explanations, code, and visualizations to provide a clear understanding of the model creation process. The latter notebook includes all of the rough work and brainstorming for the Streamlit dashboard. It is a near identical copy of `dashboard.py` and can likely be ignored.

## Conclusion

Throughout this project, I gained valuable insights into the complexities and challenges associated with predicting the stock market. While applying machine learning models such as Random Forest and LSTM showed some promising results in predicting stock movements, it became evident that accurately forecasting stock prices is an extremely difficult task. The stock market is influenced by an intricate interplay of various factors, including economic indicators, geopolitical events, company performances, and unpredictable investor sentiments. Consequently, even though the models demonstrated reasonable predictive capabilities for short-term movements, the inherent volatility and randomness in the market rendered the predictions unreliable for making financial decisions with a high degree of confidence.

As a responsible data scientist, it is crucial to acknowledge the limitations of predictive models when it comes to financial markets. While the project's performance on stock movement prediction showed improvements over random guessing, the overall accuracy was still not sufficiently high to justify betting real money on these predictions. Additionally, past performance does not guarantee future results in the stock market, making it crucial to approach such predictions with caution.
