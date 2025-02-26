import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from model import predict_stock_price  # Import LSTM model prediction function
from stocknews import StockNews

# Set page title and layout
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #56021F; /* Sidebar background color */
            color: black; /* Default text color */
        }
       
        [data-testid="stSidebar"] .stTextInput, 
        [data-testid="stSidebar"] .stDateInput label, 
        [data-testid="stSidebar"] .stDateInput input {
            color: #D17D98; /* Input text color */
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color:#F4CCE9; /* Header text color */
        }
         .stApp {
        background-color: #F4CCE9;
    }
    .stDataFrame {
            border: 1px solid #56021F;
            border-radius: 10px;
            background-color: white;
            color: black;
        }
    .stText {
            font-size: 18px;
            font-weight: bold;
            color: #1f4e79;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Fetch live stock data
st.sidebar.write("Fetching stock data...")
data = yf.download(ticker, start=start_date, end=end_date)

# Display stock data
if not data.empty:
    st.write(f"### {ticker} Stock Price Data")
    st.dataframe(data.tail())

    # Plot actual stock prices
    fig = px.line(data, x=data.index, y=data["Close"].squeeze(), title=f"{ticker} Stock Price")

   
    fig.update_layout(
        xaxis_title="📅 Date",
        yaxis_title="💰 Stock Price (USD)",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="#f0f2f6",  # Match dashboard background
        plot_bgcolor="#D17D98",  # Graph background
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Predict future stock prices
    st.write("### 📈 Stock Price Prediction using LSTM")
    predicted_prices = predict_stock_price(data)

    # Display predictions
    if predicted_prices is not None:
        prediction_df = pd.DataFrame({"Date": pd.date_range(start=end_date, periods=len(predicted_prices)), "Predicted Price": predicted_prices})
        st.dataframe(prediction_df)

        # Plot predictions
        fig_pred = px.line(prediction_df, x="Date", y="Predicted Price", title="Predicted Stock Prices")

        fig_pred.update_layout(
        xaxis_title="📅 Date",
        yaxis_title="💰 Stock Price (USD)",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="#f0f2f6",  # Match dashboard background
        plot_bgcolor="#D17D98",  # Graph background
        hovermode="x unified",
    )
        st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.error("❌ No stock data available. Please check the ticker symbol or date range.")

# Fetch and Display News
st.subheader(f"📰 Latest News for {ticker}")
try:
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()

    if not df_news.empty:
        for i in range(min(10, len(df_news))):  # Ensure at least 10 articles exist
            st.subheader(f"News {i+1}: {df_news['title'][i]}")
            st.write(f"📅 Published: {df_news['published'][i]}")
            st.write(f"📝 Summary: {df_news['summary'][i]}")
            st.write(f"🔹 Title Sentiment: {df_news['sentiment_title'][i]}")
            st.write(f"🔹 News Sentiment: {df_news['sentiment_summary'][i]}")

            # Ensure the 'link' column exists and is not empty
            if 'link' in df_news.columns and pd.notna(df_news['link'][i]):
                st.markdown(f"[Read more]({df_news['link'][i]})")
            
            st.write("---")  # Divider

    else:
        st.warning("⚠ No news articles found for this ticker.")

except Exception as e:
    st.error(f"Error fetching news: {str(e)}")

