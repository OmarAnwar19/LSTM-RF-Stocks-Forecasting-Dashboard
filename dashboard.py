# %% [markdown]
# ### Imports

# %%
from alpha_vantage.fundamentaldata import FundamentalData
import os
import streamlit as st
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from stocknews import StockNews
from datetime import date, datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

# %%
load_dotenv()

# %% [markdown]
# ### Set-Up

# %%
st.title("Market Dashboard")
ticker = st.text_input("Enter Stock Ticker")
n_years = st.slider("Data Range (years):", 1, 10)
today_date = date.today().strftime("%Y-%m-%d")
start_date = datetime.now() - timedelta(days=n_years * 365)

# %% [markdown]
# ### Load Ticker Data

# %%


def load_data():
    with st.spinner("Loading data..."):
        data = yf.download(ticker, start_date, today_date)
        data.reset_index(inplace=True)
    return data

# %% [markdown]
# ### Stock Overview

# %%


def plot_ticker_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Open"], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Close"], name="Stock Close"))
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Adj Close"], name="Adj. Close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# %%


def get_metrics(data_copy):
    annual_return = data_copy["% Change"].mean()*252*100  # 252 is trading days
    st_dev = np.std(data_copy["% Change"])*np.sqrt(252)
    st.metric(label="Annual Return",
              value=f"{round(annual_return, 4)}%")
    st.metric(label="Standard Deviation",
              value=f"{round(st_dev, 4)}%")
    st.metric(label="Risk Adjusted Return",
              value=f"{round(annual_return/(st_dev*100), 4)}%")

# %%


def current():
    if not ticker:
        st.warning("Stock Ticker Empty", icon="⚠️")
        return
    data = load_data()
    if data.empty:
        st.error("Invalid Ticker", icon="🚨")
        return

    # plot ticker data
    st.subheader("Time Series Chart")
    plot_ticker_data(data)

    # plot stock data
    st.subheader("Stock Analysis")
    data_copy = data
    data_copy["% Change"] = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    data_copy.dropna(inplace=True)
    st.write(data_copy)

    st.divider()
    get_metrics(data_copy)


# %% [markdown]
# ### Stock Data Information

# %%

@st.cache_resource
def get_balance_sheet(_fd):
    balance_sheet = _fd.get_balance_sheet_annual(ticker)[0]
    if balance_sheet.empty:
        st.error("Invalid Ticker", icon="🚨")
        return
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    return bs

# %%


@st.cache_resource
def get_income_statement(_fd):
    income_statement = _fd.get_income_statement_annual(ticker)[0]
    if income_statement.empty:
        st.error("Invalid Ticker", icon="🚨")
        return
    ics = income_statement.T[2:]
    return ics

# %%


@st.cache_data
def get_cashflow_statement(_fd):
    cash_flow = _fd.get_cash_flow_annual(ticker)[0]
    if cash_flow.empty:
        st.error("Invalid Ticker", icon="🚨")
        return
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    return cf


# %%


def fundamental():
    if not ticker:
        st.warning("Stock Ticker Empty", icon="⚠️")
        return

    # load api
    key = os.getenv("ALPHAVANTAGE_KEY")
    fd = FundamentalData(key, output_format="pandas")
    if not fd:
        st.error("API Limit Exceeded", icon="🚨")
        return

    # accounting data
    st.subheader("Balance Sheet")
    with st.spinner("Loading data..."):
        bs = get_balance_sheet(fd)
        st.write(bs)
    st.subheader("Income Statement")
    with st.spinner("Loading data..."):
        ics = get_income_statement(fd)
        st.write(ics)
    st.subheader("Cash Flow Statement")
    with st.spinner("Loading data..."):
        cf = get_cashflow_statement(fd)
        st.write(cf)

# %% [markdown]
# ### Stock Forecast Information

# %% [markdown]
# #### Training Model

# %%


def train_model(model, df_train):
    model.fit(df_train)
    future = model.make_future_dataframe(periods=n_years*365)
    forecast = model.predict(future)
    return forecast

# %%


def plot_forecast_data(model, data):
    st.subheader("Forecast Chart")
    fig1 = plot_plotly(model, data)
    fig1.layout.update(width=700, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(data)
    st.write(fig2)

# %%


def forecast():
    if not ticker:
        st.warning("Stock Ticker Empty", icon="⚠️")
        return
    data = load_data()
    if data.empty:
        st.error("Invalid Ticker", icon="🚨")
        return

    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()

    with st.spinner("Loading data..."):
        forecast_data = train_model(model, df_train)

    # plot forecast data and display analysis
    plot_forecast_data(model, forecast_data)
    st.subheader("Forecast Analysis")
    st.write(forecast_data)


# %% [markdown]
# ### Stock News

# %%
def news():
    if not ticker:
        st.warning("Stock Ticker Empty", icon="⚠️")
        return

    st.header(f"{ticker} News")
    with st.spinner("Loading data..."):
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()

    for i in range(10):
        st.divider()
        st.subheader(f"**{df_news['title'][i]}**")
        st.markdown(f"_**Published:**_ {df_news['published'][i]}")
        st.markdown(f"_{df_news['summary'][i]}_")
        ttl_sentiment = df_news["sentiment_title"][i]
        body_sentiment = df_news["sentiment_summary"][i]
        st.markdown(
            f"_**Sentiment:**_ :green[title={ttl_sentiment}, body={body_sentiment}]")

# %% [markdown]
# ### Show Stock Tabs


# %%
tab1, tab2, tab3, tab4 = st.tabs(
    ["Stock Overview", "Stock Forecast", "Accounting Data", "Market News"])
with tab1:
    current()
with tab2:
    forecast()
with tab3:
    fundamental()
with tab4:
    news()
