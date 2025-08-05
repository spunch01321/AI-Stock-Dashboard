import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline
import requests
import openai

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis")

st.title("📊 AI Stock Analyzer with News & Charts")
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if ticker:
    st.subheader(f"Stock Info: {ticker}")
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval="1d")

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"{ticker} - Last 30 Days", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Highs/Lows
    st.metric("📈 High", f"${df['High'].max():.2f}")
    st.metric("📉 Low", f"${df['Low'].min():.2f}")
    st.metric("📊 Volume (Avg)", f"{df['Volume'].mean():,.0f}")

    # News + Sentiment
    st.subheader("📰 Recent News + Sentiment")
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_NEWSAPI_KEY"
    news_data = requests.get(news_url).json()

    if "articles" in news_data:
        for article in news_data["articles"][:5]:
            title = article["title"]
            url = article["url"]
            sentiment = sentiment_model(title)[0]
            st.markdown(f"**[{title}]({url})**")
            st.write(f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
            st.write("---")
    else:
        st.warning("No news articles found or API limit reached.")