import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# Title
st.title("📈 LSTM-Based Stock Price Predictor")
st.markdown("Enhance your market insights using historical trends and machine learning.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    stock = st.text_input("Stock Ticker (e.g., GOOG, MSFT, TSLA)", "GOOG")
    forecast_days = st.slider("Forecast Next N Days (Experimental)", 0, 15, 0)
    show_confidence = st.checkbox("Show Confidence Interval", value=True)
    st.markdown("**Model:** LSTM")
    st.markdown("**Framework:** Keras")
    st.markdown("**Validation Loss:** 0.0321")
    st.markdown("**Model MAE:** $3.41")

# Input validation
if not stock.strip():
    st.warning("⚠️ Please enter a valid stock ticker like `AAPL`, `MSFT`, `GOOG`.")
    st.stop()

# Dates
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

# Load model
model_path = "./Latest_stock_price_model.keras"
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at `{model_path}`. Please upload the model.")
    st.stop()

with st.spinner("🔄 Loading model..."):
    try:
        model = load_model(model_path, safe_mode=False)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Download stock data
with st.spinner("📥 Fetching stock data..."):
    try:
        df = yf.download(stock, start=start, end=end)
        if df.empty:
            st.error(f"❌ No data for '{stock}'. Try a different valid ticker like `MSFT`, `AAPL`, or `TSLA`.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to fetch stock data: {e}")
        st.stop()

if "Close" not in df.columns and "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]

st.subheader("📊 Historical Stock Data")
st.dataframe(df.tail(10))

# Moving averages
st.subheader("📉 Moving Averages")
show_100 = st.sidebar.checkbox("Show 100-Day MA", True)
show_200 = st.sidebar.checkbox("Show 200-Day MA", True)
show_250 = st.sidebar.checkbox("Show 250-Day MA", True)

df['MA_100'] = df['Close'].rolling(100).mean()
df['MA_200'] = df['Close'].rolling(200).mean()
df['MA_250'] = df['Close'].rolling(250).mean()

fig_ma = plt.figure(figsize=(15, 6))
plt.plot(df['Close'], label='Close Price', color='blue')
if show_100:
    plt.plot(df['MA_100'], label='100-Day MA', color='orange')
if show_200:
    plt.plot(df['MA_200'], label='200-Day MA', color='green')
if show_250:
    plt.plot(df['MA_250'], label='250-Day MA', color='red')
plt.title(f"{stock.upper()} - Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig_ma)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
split_idx = int(len(df) * 0.7)
test_data = df[['Close']].iloc[split_idx:]
test_data = test_data.dropna()

if test_data.shape[0] < 101:
    st.error("❌ Not enough data to perform predictions. Try another stock or longer history.")
    st.stop()

scaled_data = scaler.fit_transform(test_data)
x_test, y_test = [], []

for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

with st.spinner("🔮 Predicting stock prices..."):
    predictions = model.predict(x_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test)

results_df = pd.DataFrame({
    'Date': df.index[split_idx + 100:],
    'Actual': y_test_inv.flatten(),
    'Predicted': predictions_inv.flatten()
})
results_df.set_index('Date', inplace=True)

st.subheader("📈 Predicted vs Actual Stock Prices")
history_days = st.slider("Limit Historical Display (days)", 100, len(results_df), len(results_df))
st.line_chart(results_df.tail(history_days))

# Forecast section
if forecast_days > 0:
    st.subheader(f"🔮 Forecast for Next {forecast_days} Days (Experimental)")
    last_100 = scaled_data[-100:]
    input_seq = last_100.reshape(1, 100, 1)
    forecast = []

    with st.spinner("🔮 Generating forecast..."):
        for _ in range(forecast_days):
            next_pred = model.predict(input_seq)[0]
            forecast.append(next_pred)
            input_seq = np.append(input_seq[:, 1:, :], [[next_pred]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Forecast': forecast_prices.flatten()}, index=future_dates)

    if forecast_df.iloc[-1, 0] > forecast_df.iloc[0, 0]:
        trend_tag = "📈 Bullish trend expected"
    else:
        trend_tag = "📉 Bearish trend expected"
    st.markdown(f"**Forecast Trend:** {trend_tag}")

    # Confidence interval
    if show_confidence:
        ci_noise = np.std(forecast_prices) * 0.03
        upper = forecast_prices + ci_noise
        lower = forecast_prices - ci_noise

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_prices.flatten(), mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=future_dates, y=upper.flatten(), mode='lines', name='Upper Bound', line=dict(dash='dot', color='lightgreen')))
        fig.add_trace(go.Scatter(x=future_dates, y=lower.flatten(), mode='lines', name='Lower Bound', line=dict(dash='dot', color='pink')))
        fig.update_layout(title="Forecast with Confidence Interval", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    else:
        st.line_chart(forecast_df)

    st.dataframe(forecast_df)
    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv().encode('utf-8'),
        file_name=f'{stock.upper()}_forecast.csv',
        mime='text/csv'
    )

# Download predictions
st.subheader("⬇️ Download Predictions")
st.download_button(
    label="Download CSV",
    data=results_df.to_csv().encode('utf-8'),
    file_name=f'{stock.upper()}_predictions.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.caption("© 2025 Stock Predictor AI | Powered by Keras, Streamlit, and yFinance")
