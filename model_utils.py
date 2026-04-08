import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def preprocess_and_train(df):
    """
    Dynamically preprocesses a dataframe and trains a Random Forest Model.
    Enforces a strict 10-feature structure to prevent mismatch errors.
    """
    df = df.copy()
    
    # 5. Safe Data Handling: Strip column names
    df.columns = df.columns.str.strip()
    
    # 1. Remove leakages
    leak_cols = ['TotalIncome', 'LoanIncomeRatio', 'EMI', 'StabilityScore', 'DebtStress']
    df = df.drop(columns=[col for col in leak_cols if col in df.columns], errors='ignore')
    
    if 'Loan_Status' in df.columns:
        target_col = 'Loan_Status'
    else:
        target_col = 'Loan_Status_Synthetic'
        df[target_col] = generate_synthetic_target(df)
        
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # 1. Enforce Fixed Feature Set
    base_features = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
    ]
    
    numeric_base = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    categorical_base = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
    
    # 2. Add Missing Columns Automatically
    for bf in base_features:
        if bf not in X_raw.columns:
            if bf in numeric_base:
                X_raw[bf] = 0.0
            else:
                X_raw[bf] = 'Unknown'
                
    # 3. Force Column Order (Removes dynamic partial selection)
    X_raw = X_raw[base_features]
    
    # Handle missing values purely over the fixed set
    for col in numeric_base:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
        med = X_raw[col].median()
        if pd.isna(med): med = 0.0
        X_raw[col] = X_raw[col].fillna(med)
        
    for col in categorical_base:
        mode_val = X_raw[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
        X_raw[col] = X_raw[col].fillna(fill_val).astype(str)
        
    # Encode categorical features
    label_encoders = {}
    X_encoded = X_raw.copy()
    for col in categorical_base:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
        
    if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        target_encoder = None
        y = y_raw.fillna(0).values

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    cm = confusion_matrix(y_test, test_preds)
    if cm.shape == (1, 1):
        padded_cm = np.zeros((2, 2), dtype=int)
        padded_cm[0, 0] = cm[0, 0]
        cm = padded_cm
        
    clf_report = classification_report(y_test, test_preds, output_dict=True)
    
    return model, label_encoders, target_encoder, base_features, train_acc, test_acc, cm, clf_report, X_encoded, y

def generate_synthetic_target(df):
    np.random.seed(42)
    scores = np.zeros(len(df))
    
    if 'Credit_History' in df.columns:
        scores += df['Credit_History'].apply(lambda x: 1.5 if str(x) in ['1', '1.0', 'Yes', 'Y'] else -1.5)
        
    inc_col = 'ApplicantIncome' if 'ApplicantIncome' in df.columns else None
    if inc_col:
        median_inc = pd.to_numeric(df[inc_col], errors='coerce').median()
        scores += pd.to_numeric(df[inc_col], errors='coerce').apply(lambda x: 0.8 if x > median_inc else -0.4)
        
    if 'Dependents' in df.columns:
        scores += df['Dependents'].apply(lambda x: -0.5 if str(x) in ['3+', '3', '4', '4+'] else 0.5)
        
    noise = np.random.uniform(-1.0, 1.0, size=len(df))
    scores += noise
    
    return (scores > 0).astype(int)

def predict_probability(model, label_encoders, feature_names, input_data):
    """
    Safely predicts probabilities enforcing strict 10 feature count and ordering.
    """
    df_input = pd.DataFrame([input_data])
    df_input.columns = df_input.columns.str.strip()
    
    numeric_base = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # 6. Prediction Safety: Guarantee all exact feature names exist
    for bf in feature_names:
        if bf not in df_input.columns:
            if bf in numeric_base:
                df_input[bf] = 0.0
            else:
                df_input[bf] = 'Unknown'
                
    # 6. Prediction Safety: Match absolute column order
    df_input = df_input[feature_names]
    
    for col, le in label_encoders.items():
        try:
            val = str(df_input[col].iloc[0])
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                df_input[col] = 0
        except Exception:
            df_input[col] = 0
                
    df_input = df_input.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    probs = model.predict_proba(df_input)[0]
    if len(probs) > 1:
        return probs[1]
    return probs[0]
