import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Switched to Classifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

# --- 1. DATA LOADING AND PREPARATION (Using credit_risk_dataset.csv) ---

FILE_NAME = 'credit_risk_dataset.csv'
TARGET_COLUMN = 'loan_status' 
NUM_SAMPLES = 20000 # We will limit the sample size for speed, but the file has more rows

@st.cache_data(show_spinner=False)
def load_and_clean_data(file_name):
    """Loads, cleans, and prepares the Credit Risk data for classification."""
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_name}' not found. Please upload it or ensure the name is correct.")
        st.stop()
        
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # 1. Feature Selection & Renaming to match prediction purpose
    # We use 'loan_status' (0=Not Defaulted, 1=Defaulted) as the Financial Risk/Stability target
    
    FEATURES = [
        'person_age', 'person_income', 'person_home_ownership', 
        'person_emp_length', 'loan_intent', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income'
    ]
    
    # Keep only required columns and the target
    df = df[FEATURES + [TARGET_COLUMN]].copy()
    
    # 2. Basic Cleaning and Imputation
    df = df.dropna(subset=['person_age', 'person_income', TARGET_COLUMN])
    
    # Clean up and limit employee length to avoid outliers/noise (e.g., max 40 years)
    df['person_emp_length'] = df['person_emp_length'].clip(upper=40) 
    
    # Limit rows to speed up demonstration
    df = df.sample(n=NUM_SAMPLES, random_state=42)
    
    return df, FEATURES

# --- 2. MODEL TRAINING AND CACHING ---
@st.cache_resource
def train_and_cache_model(df, features, target_column):
    
    X = df[features]
    y = df[target_column]

    # Define feature types based on the new dataset
    numerical_features = ['person_age', 'person_income', 'person_emp_length', 
                          'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    
    categorical_features = ['person_home_ownership', 'loan_intent']
    
    # Create Preprocessing Pipelines
    numerical_pipeline = Pipeline(steps=[
        # Impute NaNs with the mean for numerical columns
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features), 
            ('cat', categorical_pipeline, categorical_features) 
        ],
        remainder='passthrough'
    )

    # Use a Classifier as the target is now binary (0 or 1)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])

    model.fit(X, y)

    return model, features

# Load and prepare the data
try:
    data_df, model_features = load_and_clean_data(FILE_NAME)
    model, model_features = train_and_cache_model(data_df, model_features, TARGET_COLUMN)
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()
    
# Get unique categorical values for input boxes
home_options = data_df['person_home_ownership'].unique().tolist()
intent_options = data_df['loan_intent'].unique().tolist()

# --- 3. APP CONFIGURATION AND STYLING ---

st.set_page_config(page_title="AI Robo-Advisor: Financial Risk & Stability", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #007bff; /* Blue */
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {background-color: #0056b3;}
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 AI-Powered Financial Stability Advisor")
st.write("Predict your financial stability risk based on detailed income and debt metrics.")

# --- 4. USER INPUT FORM ---

with st.form("risk_advisor_form", clear_on_submit=False):
    st.subheader("Your Profile & Debt Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years):", min_value=18, max_value=80, value=35, key='age')
        income = st.number_input("Annual Gross Income ($):", min_value=10000, value=85000, step=5000, key='income')
        emp_length = st.number_input("Employment History (Years):", min_value=0.0, max_value=40.0, value=5.0, step=0.5, key='emp_length')
    
    with col2:
        home_ownership = st.selectbox("Home Ownership Status:", home_options, index=1, key='home')
        loan_intent = st.selectbox("Purpose of Loan/Debt:", intent_options, index=3, key='intent')
        loan_amnt = st.number_input("Total Debt Amount ($):", min_value=0, value=15000, step=1000, key='loan_amnt')
    
    with col3:
        loan_int_rate = st.number_input("Interest Rate on Debt (%):", min_value=0.0, max_value=30.0, value=12.5, step=0.1, format="%.1f", key='int_rate')
        # Calculate DTI based on input
        loan_percent_income = loan_amnt / income if income > 0 else 0
        st.metric(label="Calculated Debt-to-Income (DTI) Ratio:", 
                  value=f"{loan_percent_income:.2f}",
                  help="Total Debt Amount divided by Annual Income.")
        
    st.markdown("---")
    submitted = st.form_submit_button("Predict Financial Risk Score")


# --- 5. PREDICTION AND DISPLAY ---
if submitted:
    
    user_data = pd.DataFrame([{
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income
    }])
    
    # Ensure columns match the model's feature order
    user_data = user_data[model_features]

    with st.spinner("Assessing financial stability and predicting risk score..."):
        
        try:
            # Predict probability of belonging to each class (0=Stable, 1=Risky)
            probabilities = model.predict_proba(user_data)[0]
            
            # Probability of Stability (Class 0)
            stability_prob = probabilities[model.classes_ == 0][0] 
            # Probability of Risk (Class 1)
            risk_prob = probabilities[model.classes_ == 1][0]
            
            # Determine the recommended strategy
            if stability_prob >= 0.85:
                # Very High Stability
                strategy = "Aggressive Growth"
                advice = "Your profile shows very high financial stability and low risk indicators. You have significant capacity for **aggressive growth investments** (e.g., high-equity mutual funds, direct stocks). Focus on maximizing returns."
                color = "#4CAF50" # Green
            elif stability_prob >= 0.60:
                # Moderate Stability
                strategy = "Balanced Investment"
                advice = "Your profile is stable but has moderate risk factors. A **balanced portfolio** (e.g., 60% Equity, 40% Debt/Bonds) is recommended. Prioritize paying down high-interest debt (above 10%) before increasing your equity exposure."
                color = "#FFC107" # Yellow
            else:
                # Low Stability / High Risk
                strategy = "Debt Reduction & Capital Preservation"
                advice = "Your profile indicates elevated financial risk. Immediately prioritize building a large **emergency fund** (6+ months expenses) and aggressively **reducing high-interest debt**. Your investment focus should be on **capital preservation** (e.g., fixed deposits, short-term bonds)."
                color = "#F44336" # Red

            st.markdown("---")
            
            st.subheader("✅ Personalized Financial Strategy Analysis") # Changed title
            
            # Display Prediction
            col_risk, col_summary = st.columns([1, 2])
            
            with col_risk:
                st.markdown(f"""
                    <div class="container" style='border: 3px solid {color}; text-align:center;'>
                        <p style='font-size:18px; color: #333; font-weight: bold;'>Predicted Strategy Confidence</p>
                        <h1 style='font-size:40px; color: {color};'>{stability_prob * 100:.1f}%</h1>
                        <p>(Confidence in **Low Risk Profile**: {stability_prob * 100:.1f}%)</p>
                        <p>(Confidence in **High Risk Profile**: {risk_prob * 100:.1f}%)</p>
                    </div>
                """, unsafe_allow_html=True)

            with col_summary:
                st.success(f"**Recommended Strategy: {strategy}**")
                st.markdown(f"""
                    <div class="container" style='border-left: 5px solid {color};'>
                        <p style='font-size:16px;'>{advice}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            st.subheader("📊 Key Risk Factor Snapshot")
            
            st.metric(label="Debt-to-Income (DTI) Ratio", 
                      value=f"{loan_percent_income:.2f}", 
                      help="A DTI over 0.35 is generally considered high risk by many lenders.")
            st.metric(label="Annual Debt Amount", 
                      value=f"${loan_amnt:,.0f}")
            
            # --- UPDATED: Confidence Graph Labels ---
            st.subheader("Model Confidence Breakdown: Profile Prediction")
            
            # Create DataFrame for the chart
            prob_df = pd.DataFrame({
                # Labels are now clearer and tied to the strategy/profile
                'Predicted Profile Type': ['Low Risk/Stable Profile', 'High Risk/Risky Profile'], 
                'Confidence': [stability_prob, risk_prob]
            })
            
            st.bar_chart(prob_df, x='Predicted Profile Type', y='Confidence')
            # --- END UPDATED ---
            
            st.success("Analysis Complete!")
            
            # --- ADDED: Balloons Animation ---
            st.balloons()
            # --- END ADDED ---
            
        except Exception as e:
            st.error(f"Prediction error: Could not process input. Please ensure all number fields are valid. Detailed error: {e}")
            
