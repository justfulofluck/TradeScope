# 📈 TradeScope

**TradeScope** is a real-time machine learning-based trading signal system that analyzes historical price data to generate precise and explainable trade recommendations. Built with **Streamlit**, it offers an intuitive dashboard to visualize trade signals, model performance, and evaluate new market data.

---
## 🚀 New Features Implemented 

- 📡 Real-Time Market Data: Added a yfinance API integration Users can enter ticker symbols like AAPL, MSFT, TSLA
- 🧠 Live Signal Prediction: When model is trained and live feed is enabled, last row of live data is used to predict a signal in real-time
- 🔁 📊 Stock Chart: Line chart showing last 60 minutes of price data from live feed
- 🧠 🔒 Conditional Visibility: Live data, charts, and signal predictions only appear if the ticker and model are both valid No unnecessary output if no API/ticker is used
- 🎯 Save live trade log

## 🚀 Features

- 🎯 ML-powered **Buy/Sell/Hold** prediction from 6 price columns
- 📊 Beautiful and interactive **dashboard layout**
- 🔁 **Adaptable** to unseen/live data rows
- 🧠 **Confidence scoring** and trade rationale
- 📉 Risk-controlled outputs with **stop-loss** and **take-profit**
- 🐳 **Docker-ready** for easy deployment

---

## 🧠 Use Case

This tool is ideal for:

- Independent traders seeking predictive signal tools
- Quants and data scientists exploring ML in financial markets
- Fintech prototypes for trade intelligence dashboards
- Live-trading signal generation (with minimal modifications)

---

## 🧩 Input Format

Upload a CSV with the following columns:

| open | pHigh | pLow | pClose | pMean | pOpen |
|------|-------|------|--------|-------|-------|
| ...  | ...   | ...  | ...    | ...   | ...   |

Each row represents previous-period (lagged) price metrics.

---

## 🧪 How It Works

1. Preprocesses and engineers features like price range, body, and mean difference
2. Trains a **Random Forest Classifier** on labeled price direction (LONG, SHORT, HOLD)
3. Visualizes:
   - Confusion Matrix
   - Signal Distribution
4. Predicts future direction with:
   - **Trade direction**
   - **Entry price**
   - **Stop loss / Take profit**
   - **Confidence score**
   - **Rationale explanation**

---

## 💻 Run Locally

### 🐍 Option 1: Local (Python/Streamlit)
```bash
git clone https://github.com/your-username/tradescope.git
cd tradescope
pip install -r requirements.txt
streamlit run app.py
```

### 🐳 Option 2: Docker
```bash
docker build -t tradescope .
docker-compose up -d
```

---

## 📦 Deploy

### 🐳 Docker
1. Build the image:
   ```bash
   docker build -t tradescope .
   ```
2. Run the container:
   ```bash
   docker-compose up -d
   ```
3. Access the app:
   - Open a browser and go to `http://localhost:8501`
   - Upload your CSV file and see the magic!

### 📦 AWS Elastic Beanstalk
1. Create a new Elastic Beanstalk application
2. Upload your Docker image to ECR
3. Configure the environment with:
   - Instance type: Web server
   - Port: 8501
   - Health check URL: `/health`
4. Deploy!

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. For more details, please see the `License` file included in this repository.
