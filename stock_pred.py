# %% [markdown]
# ### Imports

# %%
import streamlit as st
from datetime import date, datetime, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

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
# ### Current Stock Information

# %%


def plot_ticker_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Open"], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Close"], name="Stock Close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# %%


def current():
    if not ticker:
        st.warning("Stock Ticker Empty", icon="⚠️")
        return
    data = load_data()
    if data.empty:
        st.error("Invalid Ticker", icon="🚨")
        return

    # head of fetched raw data
    st.subheader("Raw Data (first 5 cols)")
    st.write(data.head())

    # plot ticker data
    st.subheader("Time Series Data")
    plot_ticker_data(data)

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
    st.subheader("Forecast Data")
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

    # tail of forecast data
    st.subheader("Raw Data (last 5 cols)")
    st.write(forecast_data.tail())

    # plot forecast data
    plot_forecast_data(model, forecast_data)


# %% [markdown]
# ### Show Stock Tabs
# %%
tab1, tab2 = st.tabs(["Current", "Forecast"])
with tab1:
    current()
with tab2:
    forecast()
