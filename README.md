# ❤️ Heart Disease Risk Predictor

A Streamlit app that estimates heart disease risk from patient vitals and symptoms, using a Logistic Regression model trained on the UCI Heart Failure Prediction dataset.

## Features
- Interactive form for entering patient data (age, blood pressure, cholesterol, ECG results, etc.)
- Real-time risk prediction with probability score
- Clean, emoji-enhanced UI

## Model
- **Algorithm**: Logistic Regression
- **Cross-validated F1 score**: ~0.87
- **Test set recall (positive class)**: ~0.89–0.91
- Trained with a leak-free train/test split (scaling and imputation fit on training data only)

## Setup

```bash
git clone https://github.com/princegautam3612/HEART_DISEASE_PREDICTOR.git
cd HEART_DISEASE_PREDICTOR
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit frontend
- `LogReg_Heart.pkl` — trained Logistic Regression model
- `scaler.pkl` — StandardScaler fit on training data (Age, RestingBP, Cholesterol, MaxHR, Oldpeak)
- `Columns.pkl` — expected input column order after one-hot encoding
- `requirements.txt` — Python dependencies

## Disclaimer
This tool is for educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment.
