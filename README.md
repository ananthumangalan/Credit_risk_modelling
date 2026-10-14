# Credit Risk Modelling App

This project is a machine learning web application that predicts the credit risk of a loan applicant.  
The app takes customer, loan, and credit bureau-related inputs and predicts:

- Default Probability
- Credit Score
- Credit Rating

The final model is deployed using **Streamlit Community Cloud**.

## Live Demo

[Click here to open the deployed app](https://creditriskmodelling-5vtube9jytgunmdwznjqwj.streamlit.app)

## Project Overview

Credit risk modelling is used by banks and financial institutions to estimate the probability that a customer may default on a loan.

In this project, customer information, loan details, and bureau data are used to build a machine learning model that predicts whether a customer is likely to default or not.

The trained model is saved as a `.joblib` file and used inside a Streamlit web app for real-time prediction.

## Features

- Predicts loan default probability
- Calculates credit score between 300 and 900
- Assigns credit rating:
  - Poor
  - Average
  - Good
  - Excellent
- User-friendly Streamlit interface
- Deployed online using Streamlit Community Cloud

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- GitHub

## Machine Learning Workflow

The project follows these main steps:

1. Load customer, loan, and bureau datasets
2. Merge datasets using customer ID
3. Perform data cleaning
4. Create useful financial and credit-risk features
5. Handle categorical and numerical variables
6. Train a Logistic Regression model
7. Evaluate the model
8. Save the trained model using Joblib
9. Build a Streamlit app for prediction
10. Deploy the app online

## Project Structure

```text
Credit_risk_modelling/
│
├── app/
│   ├── main.py
│   ├── prediction_helper.py
│   ├── requirements.txt
│   │
│   └── artifacts/
│       └── model_data.joblib
│
├── artifacts/
│   └── model_data.joblib
│
├── dataset/
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau_data.csv
│
├── credit_risk_model.ipynb
├── requirements.txt
└── README.md
