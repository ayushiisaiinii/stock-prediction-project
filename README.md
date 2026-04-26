# Stock Prediction Model using Sentiment Analysis

## Overview

This project focuses on predicting stock market trends by combining historical data with sentiment analysis. The model analyzes daily sentiment data and uses it along with stock-related features to make predictions.

---

## Project Components

### Data

- Excel file containing daily sentiment data
- Used for training and evaluating the model

### Model

- Trained deep learning model (`.keras`)
- Built using TensorFlow/Keras
- Predicts stock trends based on sentiment + historical patterns

### Notebooks

- **sentiment_analysis.ipynb** → Performs sentiment analysis on data
- **stock_prediction_model.ipynb** → Main model training and prediction

---

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- yfinance

---

## Project Structure

```
stock-prediction-project/
│── data/
│   ├── daily_sentiment.xlsx
│
│── models/
│   ├── stock_model.keras
│
│── notebooks/
│   ├── sentiment_analysis.ipynb
│   ├── stock_prediction_model.ipynb
│
│── README.md
│── requirements.txt
```

---

## Installation & Setup

1. Clone the repository:

```
git clone https://github.com/ayushiisaiinii/stock-prediction-project.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run notebooks using Jupyter Notebook

---

## Current Status

✅ Model training completed
🚧 Frontend and backend development in progress

---

## Future Improvements

- Build interactive frontend (React / Web App)
- Develop backend API (Flask / Node.js)
- Deploy the model for real-time predictions

---
## Results

- Model used: (e.g., LSTM / Linear Regression)
- Dataset: Historical stock price data

- Note:
  Model evaluation metrics (RMSE/MAE) are not included in the current version.
  This project focuses on implementing the prediction pipeline.

## 💡 Key Highlights

- Combines sentiment analysis with stock prediction
- Uses deep learning model (.keras)
- Structured and modular project setup

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## Contact

For any queries or suggestions, feel free to reach out.
