# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:47:02 2023

@author: Rashid Alshehhi
"""

!pip install streamlit yfinance fbprophet mplfinance numpy pandas

import streamlit as st
import yfinance as yf
from fbprophet import Prophet
import mplfinance as mpf
import numpy as np
import pandas as pd

# Define the Streamlit app
st.set_page_config(page_title='Stock Prediction App', page_icon=':chart_with_upwards_trend:')

# Define the app layout
def app():
    st.title('Stock Prediction App')

    # Define the user input for the stock symbol and time range
    stock_name = st.text_input('Enter a stock symbol (e.g. AAPL)', value='AAPL')
    time_range = st.selectbox('Select a time range', ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))

    # Get the stock data from Yahoo Finance
    stock_data = yf.download(stock_name, period=time_range)

    # Define the Prophet model
    model = Prophet(daily_seasonality=True)

    # Prepare the data for Prophet
    df = pd.DataFrame(stock_data['Adj Close'])
    df.reset_index(inplace=True)
    df.columns = ['ds', 'y']

    # Fit the Prophet model to the data
    model.fit(df)

    # Define the future time range for the Prophet forecast
    future = model.make_future_dataframe(periods=365)

    # Generate the Prophet forecast
    forecast = model.predict(future)

    # Define the Pivot Point levels
    high = np.max(stock_data['High'])
    low = np.min(stock_data['Low'])
    close = stock_data['Adj Close'][-1]
    pivot = (high + low + close) / 3
    support1 = (2 * pivot) - high
    support2 = pivot - (high - low)
    resistance1 = (2 * pivot) - low
    resistance2 = pivot + (high - low)

    # Define the plot of the stock price with Pivot Points
    plot_data = stock_data.copy()
    plot_data['Pivot'] = pivot
    plot_data['S1'] = support1
    plot_data['S2'] = support2
    plot_data['R1'] = resistance1
    plot_data['R2'] = resistance2
    plot_data.index.name = 'Date'
    fig1 = mpf.plot(plot_data, type='candle', style='charles', mav=(10, 20),
                    title=f'{stock_name} ({time_range}) with Pivot Points',
                    ylabel='Price ($)', ylabel_lower='Shares\nTraded',
                    returnfig=True)

    # Define the plot of the Prophet forecast
    fig2 = model.plot(forecast).get_figure()

    # Display the plots
    st.write(fig1)
    st.write(fig2)

# Run the Streamlit app
if __name__ == '__main__':
    app()
