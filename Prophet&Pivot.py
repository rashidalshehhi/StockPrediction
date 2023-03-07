#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
from fbprophet import Prophet
import pandas as pd
from flask import Flask, render_template, request
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pivot_points import PivotPoints
import streamlit as st

st.set_page_config(page_title="Stock Price Prediction App")

@st.cache
def load_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="max")
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    return df

@st.cache
def prophet_prediction(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted"))
    fig.layout.update(title="Facebook Prophet Prediction")
    return fig

@st.cache
def pivot_points_analysis(df):
    pp = PivotPoints(df)
    pp_data = pp.get_pivot_points()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_data.index, y=pp_data["pp"], name="Pivot Point"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_data.index, y=pp_data["r1"], name="Resistance 1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_data.index, y=pp_data["s1"], name="Support 1"), row=1, col=1)
    fig.layout.update(title="Pivot Points Analysis")
    return fig

def app():
    st.title("Stock Price Prediction App")

    symbol = st.text_input("Enter stock symbol (e.g. AAPL)")
    if symbol:
        try:
            df = load_data

