import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from transformers import pipeline
import requests
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import openai

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Load API keys securely
news_key = st.secrets["api_keys"]["newsapi"]
openai.api_key = st.secrets["api_keys"]["openai"]

st.title("📊 AI Stock Analyzer with Projections, News & GPT Summaries")

# User inputs
ticker = st.text_input("Enter Stock Ticker:", "AAPL")
period = st.selectbox("Select Time Frame", ["1mo", "3mo", "6mo", "1y"], index=0)

if ticker:
    st.subheader(f"Stock Info: {ticker} - {period}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval="1d")

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"{ticker} - {period} Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Highs/Lows
    st.metric("📈 High", f"${df['High'].max():.2f}")
    st.metric("📉 Low", f"${df['Low'].min():.2f}")
    st.metric("📊 Volume (Avg)", f"{df['Volume'].mean():,.0f}")

    # Trend projection
    st.subheader("📈 Projected Trend (Linear Regression)")
    df = df.reset_index()
    df['DateInt'] = df['Date'].map(datetime.toordinal)
    df.dropna(inplace=True)
    X = df[['DateInt']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Project next 7 days
    last_date = df['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    future_date_ints = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_prices = model.predict(future_date_ints)

    future_df = pd.DataFrame({'Date': future_dates, 'Projected Close': future_prices})

    # Merge for plotting
    combined_df = pd.concat([df[['Date', 'Close']], future_df.rename(columns={'Projected Close': 'Close'})], ignore_index=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Close'], name='Price & Projection'))
    fig2.update_layout(title="📉 Historical + Projected Closing Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)

    # News + GPT Summary
    st.subheader("📰 Recent News + GPT Summary + Sentiment")
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={news_key}"
    news_data = requests.get(news_url).json()

    if "articles" in news_data:
        for article in news_data["articles"]:
            title = article.get("title", "")
            url = article.get("url", "")
            description = article.get("description", "")
            content = article.get("content", "")

            # Run sentiment
            sentiment = sentiment_model(title)[0]

            # GPT summary
            try:
                prompt = (
                    f"Summarize the following stock market news article in 2-3 sentences:\n\n"
                    f"Title: {title}\n\n"
                    f"Content: {description or content}"
                )
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response["choices"][0]["message"]["content"]
            except Exception as e:
                summary = "Summary unavailable due to an error."

            st.markdown(f"### [{title}]({url})")
            st.write(f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})")
            st.write(f"**GPT Summary:** {summary}")
            st.write("---")
    else:
        st.warning("No news articles found or API limit reached.")
