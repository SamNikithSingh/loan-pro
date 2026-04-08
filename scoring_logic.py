import pandas as pd
import numpy as np

def calculate_rule_based_score(inputs):
    """
    Calculates a rule-based score safely checking `.get()` against unpredictable datasets.
    """
    score = 0.0
    
    # 1. Credit History (40%)
    try:
        ch = inputs['Credit_History']
    except KeyError:
        ch = None
        
    if str(ch) in ['1', '1.0', 'Yes', 'Y']:
        score += 0.4
    elif ch is None or pd.isna(ch):
        score += 0.2
        
    # Income aggregation safely
    try: app_inc = pd.to_numeric(inputs['ApplicantIncome'], errors='coerce')
    except KeyError: app_inc = 0
    try: co_inc = pd.to_numeric(inputs['CoapplicantIncome'], errors='coerce')
    except KeyError: co_inc = 0
    
    app_inc = 0 if pd.isna(app_inc) else app_inc
    co_inc = 0 if pd.isna(co_inc) else co_inc
    
    total_income = app_inc + co_inc
    if total_income == 0:
        try:
            total_income = pd.to_numeric(inputs['TotalIncome'], errors='coerce')
        except KeyError:
            total_income = 0
        total_income = 0 if pd.isna(total_income) else total_income
        
    try:
        loan_amount = pd.to_numeric(inputs['LoanAmount'], errors='coerce')
    except KeyError:
        loan_amount = 0
    loan_amount = 0 if pd.isna(loan_amount) else loan_amount
    
    # 2. Loan Income Ratio (20%)
    if total_income > 0 and loan_amount > 0:
        ratio = loan_amount / total_income
        if ratio < 0.2:
            score += 0.2
        elif ratio < 0.4:
            score += 0.15
        elif ratio < 0.6:
            score += 0.1
        else:
            score += 0.05
    else:
        score += 0.1
    
    # 3. Total Income (20%)
    if total_income > 10000:
        score += 0.2
    elif total_income > 5000:
        score += 0.15
    elif total_income > 3000:
        score += 0.1
    elif total_income > 0:
        score += 0.05
    else:
        score += 0.1
        
    # 4. Dependents (10%)
    dep = str(inputs.get('Dependents', '0')).replace('+', '')
    if dep.isdigit():
        dep_val = int(dep)
        if dep_val == 0:
            score += 0.1
        elif dep_val == 1:
            score += 0.08
        elif dep_val == 2:
            score += 0.06
        else:
            score += 0.04
    else:
        score += 0.07
        
    # 5. Education (10%)
    edu = str(inputs.get('Education', 'Unknown')).lower()
    if 'grad' in edu and 'not' not in edu:
        score += 0.1
    else:
        score += 0.07
        
    return min(score, 1.0)

def generate_alerts(inputs):
    """
    Generates risk alerts dynamically safely.
    """
    alerts = []
    
    app_inc = pd.to_numeric(inputs.get('ApplicantIncome', 0), errors='coerce')
    co_inc = pd.to_numeric(inputs.get('CoapplicantIncome', 0), errors='coerce')
    app_inc = 0 if pd.isna(app_inc) else app_inc
    co_inc = 0 if pd.isna(co_inc) else co_inc
    
    total_income = app_inc + co_inc
    if total_income == 0:
        total_income = pd.to_numeric(inputs.get('TotalIncome', 0), errors='coerce')
        total_income = 0 if pd.isna(total_income) else total_income

    loan_amount = pd.to_numeric(inputs.get('LoanAmount', 0), errors='coerce')
    loan_amount = 0 if pd.isna(loan_amount) else loan_amount
    
    ratio = loan_amount / total_income if total_income > 0 else 0
    if ratio > 0.5:
        alerts.append("⚠️ High Loan-to-Income Ratio (>0.5)")
        
    ch = str(inputs.get('Credit_History', 'Unknown'))
    if ch in ['0', '0.0', 'No', 'N']:
        alerts.append("⚠️ Poor Credit History")
        
    if 0 < total_income < 4000:
        alerts.append("⚠️ Low Total Monthly Income (<4000)")
        
    dep = str(inputs.get('Dependents', '0')).replace('+', '')
    if dep.isdigit() and int(dep) >= 3:
        alerts.append("⚠️ High number of Dependents (≥3)")
        
    return alerts

def get_risk_category(score):
    if score >= 750:
        return "Excellent", "green"
    elif score >= 650:
        return "Good", "blue"
    elif score >= 550:
        return "Moderate", "orange"
    else:
        return "High Risk", "red"

def calculate_final_credit_score(rule_score, ml_prob):
    combined_score = (0.7 * rule_score) + (0.3 * ml_prob)
    return int(300 + (combined_score * 600))
