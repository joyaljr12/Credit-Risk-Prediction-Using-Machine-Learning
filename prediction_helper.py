import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import minmax_scale


#path to the saved model and scaler
model_path = 'artifacts/model_data.joblib'

# Load the model and scaler
model_data = joblib.load(model_path)
model = model_data['model']
features = model_data['features']
scaler = model_data['scaler']
col_to_scale = model_data['col_to_scale']





def prepare_data(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency, delinquent_to_loan_month_ratio, 
            credit_utilization_ratio, num_open_accounts, residence_type, loan_purpose, loan_type):
    # Create a DataFrame with the input values
    input_data = {
        'age': age,
        'credit_utilization_ratio': credit_utilization_ratio,
        'delinquent_to_loan_month_ratio': delinquent_to_loan_month_ratio,
        'loan_to_income_ratio': loan_amount / income if income > 0 else 0,
        'avg_dpd_per_delinquency': avg_dpd_per_deliquency,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # additional dummy fields just for scaling purpose, becuase the model was scaled on these features
        'number_of_dependants': 1,  # Dummy value
        'years_at_current_address': 1,  # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }

    df = pd.DataFrame([input_data])

    # Scale the features
    df[col_to_scale] = scaler.transform(df[col_to_scale])

    df = df[features]  # Ensure the order of features matches the training data

    return df


    # Make the prediction
    prediction = model.predict(input_data)

   
def calculate_credit_score_and_rating(input_df, base_score=300, max_score=600):
    x= np.dot(input_df, model.coef_.T) + model.intercept_   # x=m1*x1+m2*x2+...+b
    default_probability = 1 / (1 + np.exp(-x))  # Sigmoid function
    non_default_probability = 1 - default_probability
    credit_score = base_score +  non_default_probability.flatten() * max_score  # Scale to the range [300, 900]

    def get_credit_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined' # in case the score is out of the expected range

    rating = get_credit_rating(credit_score[0])  # Get the rating for the first (and only) score in the array

    return default_probability.flatten()[0], int(credit_score[0]), rating  # Return the first (and only) values
  

   

    

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency, delinquent_to_loan_month_ratio, 
            credit_utilization_ratio, num_open_accounts, residence_type, loan_purpose, loan_type):

    input_df = prepare_data(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency, delinquent_to_loan_month_ratio, 
                            credit_utilization_ratio, num_open_accounts, residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score_and_rating(input_df)
    

    return probability, credit_score, rating


if __name__ == "__main__":
    # Example usage
    age = 30
    income = 50000
    loan_amount = 10000
    loan_tenture_months = 36
    avg_dpd_per_deliquency = 20
    delinquent_to_loan_month_ratio = 0.5
    credit_utilization_ratio = 30
    num_open_accounts = 2
    residence_type = 'Owned'
    loan_purpose = 'Education'
    loan_type = 'Unsecured'

    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenture_months, avg_dpd_per_deliquency, 
                                                delinquent_to_loan_month_ratio, credit_utilization_ratio, num_open_accounts, 
                                                residence_type, loan_purpose, loan_type)

    print(f'Probability of Default: {probability}%')
    print(f'Credit Score: {credit_score}')
    print(f'Credit Rating: {rating}')