import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
from datetime import datetime, timedelta
import time

class TradingAlgorithm:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def preprocess_data(self, data):
        required_columns = ["open", "pHigh", "pLow", "pClose", "pMean", "pOpen"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data = data.copy()
        data.dropna(inplace=True)
        data['range'] = data['pHigh'] - data['pLow']
        data['body'] = data['pClose'] - data['pOpen']
        data['mean_diff'] = data['pClose'] - data['pMean']
        data['target'] = data['pClose'].shift(-1) - data['pClose']
        data['target'] = data['target'].apply(lambda x: 1 if x > 0.0002 else (-1 if x < -0.0002 else 0))

        features = data[["open", "pHigh", "pLow", "pClose", "pMean", "pOpen", "range", "body", "mean_diff"]]
        labels = data['target']

        return features[:-1], labels[:-1]

    def train_model(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        self.trained = True
        self.X_test, self.y_test, self.y_pred = X_test, y_test, y_pred
        return report

    def generate_trade_signal(self, row):
        if not self.trained:
            raise Exception("Model is not trained yet.")
        try:
            row['range'] = row['pHigh'] - row['pLow']
            row['body'] = row['pClose'] - row['pOpen']
            row['mean_diff'] = row['pClose'] - row['pMean']
            features = row[["open", "pHigh", "pLow", "pClose", "pMean", "pOpen", "range", "body", "mean_diff"]].values.reshape(1, -1)
            prediction = self.model.predict(features)[0]
            confidence = np.max(self.model.predict_proba(features)) * 100
            direction = {1: "LONG", -1: "SHORT", 0: "HOLD"}[prediction]
            entry_price = row['pClose']
            stop_loss = entry_price - 0.0005 if direction == "LONG" else entry_price + 0.0005
            take_profit = entry_price + 0.001 if direction == "LONG" else entry_price - 0.001
            rationale = f"Signal based on price body, range, and mean difference with {confidence:.2f}% confidence."
            return {
                "trade_direction": direction,
                "entry_price": round(entry_price, 5),
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "confidence_score": round(confidence, 2),
                "rationale": rationale
            }
        except Exception as e:
            return {"error": str(e)}

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["LONG", "HOLD", "SHORT"], yticklabels=["LONG", "HOLD", "SHORT"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def plot_prediction_distribution(y_pred):
    unique, counts = np.unique(y_pred, return_counts=True)
    fig2, ax2 = plt.subplots()
    ax2.pie(counts, labels=[{1: 'LONG', 0: 'HOLD', -1: 'SHORT'}[i] for i in unique], autopct='%1.1f%%', startangle=140)
    ax2.set_title("Trade Signal Distribution")
    st.pyplot(fig2)

def fetch_yfinance_data(ticker):
    try:
        df = yf.download(ticker, period='1d', interval='1m')
        df = df.reset_index()
        df.rename(columns={
            'Open': 'open',
            'High': 'pHigh',
            'Low': 'pLow',
            'Close': 'pClose',
        }, inplace=True)
        df['pOpen'] = df['open'].shift(1)
        df['pMean'] = (df['pHigh'] + df['pLow']) / 2
        return df.dropna()
    except Exception as e:
        return None

def main():
    st.set_page_config(page_title="Trading Signal Dashboard", layout="wide")
    with st.sidebar:
        st.title("⚙️ Settings")
        uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")
        st.markdown("---")
        st.markdown("### 📡 Live Market Data")
        live_ticker = st.text_input("Search Ticker (e.g. AAPL, MSFT, TSLA)")
        show_live = st.checkbox("Enable Live Market Predictions")

    st.title("📈 Trading Intelligence Dashboard")
    algo = TradingAlgorithm()

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            with st.expander("🔍 Raw Data Preview", expanded=False):
                st.dataframe(data.head(), use_container_width=True)

            with st.spinner("⚙️ Training model on uploaded data..."):
                report = algo.train_model(data)
            st.success("✅ Model trained successfully!")

            st.markdown("### 🧠 Model Performance Summary")
            accuracy = report['accuracy'] * 100
            precision = report['weighted avg']['precision'] * 100
            recall = report['weighted avg']['recall'] * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.1f}%")
            col2.metric("Precision", f"{precision:.1f}%")
            col3.metric("Recall", f"{recall:.1f}%")

            with st.expander("📊 Model Diagnostics", expanded=True):
                plot_confusion_matrix(algo.y_test, algo.y_pred)
                plot_prediction_distribution(algo.y_pred)

            st.subheader("🔎 Try a Specific Data Row")
            index = st.number_input("Select row index to test", min_value=0, max_value=len(data)-1, value=0)
            signal = algo.generate_trade_signal(data.iloc[index])

            if "error" in signal:
                st.error(signal["error"])
            else:
                st.markdown("### 📌 Latest Trade Signal")
                colA, colB, colC = st.columns(3)
                colA.metric("Direction", signal["trade_direction"])
                colB.metric("Entry Price", signal["entry_price"])
                colC.metric("Confidence %", f"{signal['confidence_score']}%")
                colX, colY = st.columns(2)
                colX.metric("Stop Loss", signal["stop_loss"])
                colY.metric("Take Profit", signal["take_profit"])
                with st.expander("🧠 Trade Rationale"):
                    st.markdown(f"> {signal['rationale']}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    if show_live and live_ticker:
        st.markdown("---")
        st.subheader(f"📡 Live Market Feed: {live_ticker.upper()}")
        df_live = fetch_yfinance_data(live_ticker)

        if df_live is not None and not df_live.empty:
            st.line_chart(df_live.set_index("Datetime")["pClose"].tail(60))
            if algo.trained:
                signal = algo.generate_trade_signal(df_live.iloc[-1])
                if "error" in signal:
                    st.error(signal["error"])
                else:
                    st.success(f"Live Signal: {signal['trade_direction']} @ {signal['entry_price']} ({signal['confidence_score']}% confidence)")
        else:
            st.warning("No live data found or failed to fetch from API.")

if __name__ == "__main__":
    main()