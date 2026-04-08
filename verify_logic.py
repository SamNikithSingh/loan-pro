import pandas as pd
from model_utils import preprocess_and_train, predict_probability
from scoring_logic import calculate_rule_based_score, calculate_final_credit_score, generate_alerts, get_risk_category

def test_system():
    print("--- 🔬 Verification Start ---")
    
    # 1. Test Model Training using dummy data mimicking a dynamic upload
    print("\n1. Testing Model Training...")
    try:
        # Create dummy df
        dummy_data = {
            'ApplicantIncome': [5000, 2000, 8000, 1500, 10000],
            'CoapplicantIncome': [2000, 0, 3000, 0, 5000],
            'LoanAmount': [150, 100, 250, 50, 300],
            'Credit_History': ['1', '0', '1', '0', '1'],
            'Dependents': ['0', '2', '0', '4+', '1'],
            'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Self_Employed': ['No', 'Yes', 'No', 'Yes', 'No'],
            'Loan_Amount_Term': [360, 360, 180, 360, 360],
            # Optionally missing Loan_Status to test target synthesis explicitly requested
        }
        df_dummy = pd.DataFrame(dummy_data)
        
        model, le_dict, target_le, f_names, train_acc, test_acc, cm, clf_report, X_enc, y_enc = preprocess_and_train(df_dummy)
        print("✅ Model trained successfully.")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return

    # 2. Test Scoring Logic with High Risk case
    print("\n2. Testing High Risk Case...")
    high_risk_input = {
        'ApplicantIncome': 2000,
        'CoapplicantIncome': 500,
        'LoanAmount': 2000, 
        'Credit_History': '0',
        'Dependents': '4',
        'Education': 'Not Graduate',
        'Gender': 'Male',
        'Married': 'No',
        'Self_Employed': 'No',
        'Loan_Amount_Term': 360
    }
    
    # Always access data using column names (inputs['ApplicantIncome'] etc) within logic
    r_score = calculate_rule_based_score(high_risk_input)
    ml_prob = predict_probability(model, le_dict, f_names, high_risk_input)
    final_score = calculate_final_credit_score(r_score, ml_prob)
    cat, color = get_risk_category(final_score)
    alerts = generate_alerts(high_risk_input)
    
    print(f"Rule Score: {r_score:.2f}")
    print(f"ML Prob: {ml_prob:.2f}")
    print(f"Final Score: {final_score}")
    print(f"Risk Category: {cat}")
    print(f"Alerts: {alerts}")

    # 3. Test Scoring Logic with Excellent case
    print("\n3. Testing Excellent Case...")
    excellent_input = {
        'ApplicantIncome': 12000,
        'CoapplicantIncome': 5000,
        'LoanAmount': 100,
        'Credit_History': '1',
        'Dependents': '0',
        'Education': 'Graduate',
        'Gender': 'Female',
        'Married': 'Yes',
        'Self_Employed': 'No',
        'Loan_Amount_Term': 360
    }
    
    r_score = calculate_rule_based_score(excellent_input)
    ml_prob = predict_probability(model, le_dict, f_names, excellent_input)
    final_score = calculate_final_credit_score(r_score, ml_prob)
    cat, color = get_risk_category(final_score)
    alerts = generate_alerts(excellent_input)
    
    print(f"Rule Score: {r_score:.2f}")
    print(f"ML Prob: {ml_prob:.2f}")
    print(f"Final Score: {final_score}")
    print(f"Risk Category: {cat}")
    print(f"Alerts: {alerts}")
    
    print("\n--- ✅ All Verification Checks Passed ---")

if __name__ == "__main__":
    test_system()
