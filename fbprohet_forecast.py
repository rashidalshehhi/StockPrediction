import streamlit as st
from fbprophet import Prophet
from datetime import datetime, timedelta
import yfinance as yf
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


# function to load data and perform predictions
def load_data(ticker, year):
    # load data from yahoo finance
    stock_data = yf.download(ticker, start="2015-01-01", end=datetime.now())
    stock_data = stock_data.reset_index()
    stock_data = stock_data[["Date", "Close"]]
    stock_data = stock_data.rename(columns={"Date": "ds", "Close": "y"})

    # create Prophet model
    model = Prophet()
    model.fit(stock_data)

    # create future dataframe for predictions
    future = model.make_future_dataframe(periods=365)
    future = future[future["ds"].dt.year == year]

    # perform predictions
    forecast = model.predict(future)

    # plot predictions
    fig = plot_plotly(model, forecast)
    fig.update_layout(title=f"Predictions for {ticker} in {year}", xaxis_title="Date", yaxis_title="Close Price")
    fig.update_yaxes(fixedrange=True)

    return forecast, fig


# create web app interface
st.title("Stock Prediction Web App")

# get user input for stock symbol and year
ticker = st.sidebar.text_input("Enter stock symbol (e.g. AAPL for Apple):")
year = st.sidebar.slider("Enter year to predict for:", min_value=2015, max_value=datetime.now().year, step=1)

# load data and perform predictions
if ticker:
    forecast, fig = load_data(ticker, year)

    # display predictions
    st.plotly_chart(fig)

    st.write(f"Predictions for {ticker} in {year}:")
    st.write(forecast[["ds", "yhat"]].tail())


