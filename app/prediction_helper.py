import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "artifacts/model_data.joblib"


# Load saved model components once when this file is imported.
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
cols_to_scale = model_data["cols_to_scale"]


RATING_BANDS = [
    (300, 500, "Poor"),
    (500, 650, "Average"),
    (650, 750, "Good"),
    (750, 901, "Excellent"),
]


def get_rating(score):
    """Return the credit rating label for a given credit score."""
    for lower_limit, upper_limit, rating in RATING_BANDS:
        if lower_limit <= score < upper_limit:
            return rating
    return "Undefined"


def prepare_input(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type,
):
    """Create the final model-ready dataframe from user input."""

    loan_to_income = loan_amount / income if income > 0 else 0

    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,

        # One-hot encoded categorical values.
        "residence_type_Owned": 1 if residence_type == "Owned" else 0,
        "residence_type_Rented": 1 if residence_type == "Rented" else 0,
        "loan_purpose_Education": 1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home": 1 if loan_purpose == "Home" else 0,
        "loan_purpose_Personal": 1 if loan_purpose == "Personal" else 0,
        "loan_type_Unsecured": 1 if loan_type == "Unsecured" else 0,

        # Extra fields needed only because the scaler was fitted with them.
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1,
    }

    input_df = pd.DataFrame([input_data])

    # Apply the same scaling used during training.
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # Keep only the columns expected by the trained model, in the same order.
    input_df = input_df[features]

    return input_df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """Calculate default probability, credit score, and rating."""

    log_odds = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-log_odds))

    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    score = int(credit_score[0])
    rating = get_rating(score)
    probability = default_probability.flatten()[0]

    return probability, score, rating


def predict(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type,
):
    """Prepare user input and return model prediction results."""

    input_df = prepare_input(
        age,
        income,
        loan_amount,
        loan_tenure_months,
        avg_dpd_per_delinquency,
        delinquency_ratio,
        credit_utilization_ratio,
        num_open_accounts,
        residence_type,
        loan_purpose,
        loan_type,
    )

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating
