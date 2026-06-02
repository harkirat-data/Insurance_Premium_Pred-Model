# Insurance Premium Prediction

A machine learning web app that predicts insurance premium category based on user health and demographic data.

## Tech Stack
- **Backend:** FastAPI, Pydantic
- **Frontend:** Streamlit
- **ML Model:** Scikit-learn (classification)
- **Language:** Python

## Features
- REST API built with FastAPI for model serving
- Streamlit frontend for user input and predictions
- Pydantic validation with computed fields (BMI, age group, city tier)
- Predicts premium category: Low / Medium / High

## How to Run

**Backend:**
```bash
uvicorn app:app --reload
```

**Frontend:**
```bash
streamlit run frontend.py
```

## Project Structure
```
├── app.py              # FastAPI backend
├── frontend.py         # Streamlit frontend
├── model.pkl           # Trained ML model
├── insurance.csv       # Dataset
└── Insurance_Premium_prediction-Model.ipynb  # EDA + model training
```
