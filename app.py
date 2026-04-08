import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import preprocess_and_train, predict_probability
from scoring_logic import calculate_rule_based_score, calculate_final_credit_score, generate_alerts, get_risk_category

st.set_page_config(page_title="Dynamic Credit Scoring System", layout="wide")
st.title("🛡️ Dynamic Credit Scoring System")
st.markdown("Upload a CSV dataset to train a ML model dynamically and generate realistic financial analytics.")

# Sidebar
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])

if uploaded_file is not None:
    # Attempt to read the dataset
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Dataset securely loaded! Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns.")
        
        analyze_btn = st.button("🛠️ Analyze Dataset & Train Model", type="primary")
        
        if analyze_btn or 'model' in st.session_state:
            if analyze_btn:
                with st.spinner("Executing dynamic preprocessing, resolving leakage, and training generalized models..."):
                    model, le_dict, target_le, f_names, train_acc, test_acc, cm, clf_report, X_enc, y_enc = preprocess_and_train(df_raw)
                    st.session_state['model'] = model
                    st.session_state['le_dict'] = le_dict
                    st.session_state['f_names'] = f_names
                    st.session_state['train_acc'] = train_acc
                    st.session_state['test_acc'] = test_acc
                    st.session_state['cm'] = cm
                    st.session_state['clf_report'] = clf_report
                    st.session_state['X_enc'] = X_enc
                    st.session_state['df_raw'] = df_raw
            
            # Retrieving stored state
            model = st.session_state['model']
            le_dict = st.session_state['le_dict']
            f_names = st.session_state['f_names']
            train_acc = st.session_state['train_acc']
            test_acc = st.session_state['test_acc']
            cm = st.session_state['cm']
            clf_report = st.session_state['clf_report']
            X_enc = st.session_state['X_enc']
            df_raw = st.session_state['df_raw']
            
            st.markdown("---")
            st.header("📊 Global Model Diagnostics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Accuracy (Holdout)", f"{test_acc*100:.2f}%")
            with col2:
                st.metric("Train Accuracy", f"{train_acc*100:.2f}%")
            with col3:
                st.info("💡 **Why avoiding 100% is crucial:** A pure 100% score implies deterministic overfitting or *Data Leakage*. By clearing derived metrics (like LoanIncomeRatio) our random forest mirrors a generalizing real-world system (aiming ~90-95%) rather than memorizing the past.")
                
            # Dashboards
            st.subheader("Interactive Visualizations & Metrics")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Importance", "Correlation Heatmap", "Loan vs Income (Scatter)", "Confusion Matrix", "Classification Report"])
            
            with tab1:
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                importances = model.feature_importances_
                sns.barplot(x=importances, y=f_names, ax=ax1, palette='viridis', hue=f_names, legend=False)
                ax1.set_title("Which columns influence approvals the most?")
                st.pyplot(fig1)
                
            with tab2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                numeric_df = X_enc.select_dtypes(include=['number'])
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax2, linewidths=.5)
                ax2.set_title("Global Column Correlation Heatmap")
                st.pyplot(fig2)
                
            with tab3:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                if 'ApplicantIncome' in df_raw.columns and 'LoanAmount' in df_raw.columns:
                    sns.scatterplot(data=df_raw, x='ApplicantIncome', y='LoanAmount', alpha=0.6, ax=ax3, color="#0068c9")
                    ax3.set_title("Loan Request Scale vs Base Applicant Income")
                else:
                    ax3.text(0.5, 0.5, "Standard Loan/Income columns absent in uploaded CSV.", ha='center')
                    ax3.set_axis_off()
                st.pyplot(fig3)
                
            with tab4:
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
                ax4.set_xlabel("Predicted Label")
                ax4.set_ylabel("True Label")
                ax4.set_title("Machine Learning Truth Matrix")
                st.pyplot(fig4)
                
            with tab5:
                st.write("**Full Sklearn Classification Report:**")
                df_report = pd.DataFrame(clf_report).transpose()
                st.dataframe(df_report.style.format(precision=3), use_container_width=True)
                
            st.markdown("---")
            
            st.header("👤 Individual Applicant Diagnostics")
            st.write("Scan the dataset line-by-line using the selector below to visualize live inference against the robust predictive model.")
            
            row_idx = st.number_input("Select Dataset Row Index (Applicant ID)", min_value=0, max_value=len(df_raw)-1, value=0)
            selected_row = df_raw.iloc[row_idx].to_dict()
            
            with st.expander("View Raw Applicant Data Trace"):
                st.json({k: str(v) for k, v in selected_row.items() if not pd.isna(v)})
                
            score_btn = st.button("👉 Compute Hybrid Credit Score", type="secondary")
            
            if score_btn:
                with st.spinner("Processing neural combinations and heuristical metrics..."):
                    ml_prob = predict_probability(model, le_dict, f_names, selected_row)
                    rule_score = calculate_rule_based_score(selected_row)
                    
                    final_score = calculate_final_credit_score(rule_score, ml_prob)
                    cat, color = get_risk_category(final_score)
                    alerts = generate_alerts(selected_row)
                    
                    res_c1, res_c2 = st.columns([1, 1])
                    
                    with res_c1:
                        st.subheader("💡 Terminal Assessment")
                        st.markdown(f'''
                        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; text-align: center; border: 2px solid {color};">
                            <h1 style="font-size: 65px; color: {color}; margin-bottom: 0;">{final_score}</h1>
                            <p style="font-size: 26px; color: {color}; font-weight: 800; text-transform: uppercase;">{cat} RISK</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.progress(float(min(1.0, max(0.0, (final_score - 300) / 600))))
                        st.caption("Credit Score Bounds (300 to 900)")
                        
                        st.info(f"**ML Engine Confidence:** {ml_prob*100:.1f}%\n\n**Financial Rules Passing:** {rule_score*100:.1f}%")
                        
                    with res_c2:
                        st.subheader("⚠️ Automated Alerts")
                        if alerts:
                            for a in alerts:
                                st.error(a)
                        else:
                            st.success("✅ Clean financial profile. Zero automated risk warnings fired.")
                            
                        st.write("---")
                        st.write("Risk metrics evaluated securely without derived internal features mapped, allowing high generalization success mapping real-world behaviors.")

    except Exception as e:
        st.error(f"FATAL ERROR: Could not parse CSV or train model. Detailed Traceback: {e}")

else:
    st.info("👈 Upload your targeted Dataset using the sidebar panel to ignite the Dashboard.")
