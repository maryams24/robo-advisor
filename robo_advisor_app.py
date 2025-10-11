import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

# --- 1. DATA LOADING AND PREPARATION (Using credit_risk_dataset.csv) ---

FILE_NAME = 'credit_risk_dataset.csv'
# TARGET: Predicts the user's most likely Loan Intent (Multi-class: 6 categories)
TARGET_COLUMN = 'loan_intent' 
NUM_SAMPLES = 20000 

# Define advice mapped to the categories (Loan Intent) - Renamed to "Financial Profiles"
INTENT_ADVICE = {
    'DEBTCONSOLIDATION': {
        'title': "Debt Restructuring & High-Interest Payoff Strategy",
        'advice': "Your profile aligns with needing to **consolidate existing high-interest debts**. Prioritize the 'snowball' or 'avalanche' method to clear these obligations quickly. Avoid taking on new debt until your Debt-to-Income (DTI) ratio is significantly lower.",
        'color': '#F44336', # Red/Urgent
        'profile': 'High-Debt Reduction Profile' # Simplified Name
    },
    'EDUCATION': {
        'title': "Future Income Maximization & Investment Strategy",
        'advice': "Your profile suggests investments in **education or self-improvement**. Focus on optimizing your future earning potential. Treat this debt as a strategic investment, but ensure you have a clear plan for repayment based on expected post-education income.",
        'color': '#2196F3', # Blue/Growth
        'profile': 'Career Growth Profile' # Simplified Name
    },
    'HOMEIMPROVEMENT': {
        'title': "Asset Building & Large Purchase Budgeting Strategy",
        'advice': "Your profile indicates a need for **asset improvement**. Ensure your spending on home improvements doesn't compromise your emergency savings. Budget for contingency costs, as renovations often exceed initial estimates.",
        'color': '#795548', # Brown/Stability
        'profile': 'Asset Optimization Profile' # Simplified Name
    },
    'MEDICAL': {
        'title': "Emergency Fund & Health Expense Protection Strategy",
        'advice': "Your profile highlights potential for **unexpected medical expenses**. Your top priority must be establishing an ample **emergency fund (8-12 months)**. Consider reviewing health insurance coverage to minimize future out-of-pocket costs.",
        'color': '#FF9800', # Orange/Caution
        'profile': 'Protection Focus Profile' # Simplified Name
    },
    'PERSONAL': {
        'title': "Budget Review & Conservative Savings Strategy",
        'advice': "Your profile suggests generalized financial needs. Review your discretionary spending carefully. Allocate a balanced portfolio (e.g., 60/40 stocks/bonds) but **increase your monthly savings contributions** before any major new purchases.",
        'color': '#9E9E9E', # Grey/Balanced
        'profile': 'General Balance Profile' # Simplified Name
    },
    'VENTURE': {
        'title': "High-Risk, High-Reward Capital Strategy",
        'advice': "Your profile aligns with **entrepreneurial or high-risk investments**. While potential returns are high, isolate this capital from your personal finances. Ensure your core retirement and emergency funds are fully protected and liquid.",
        'color': '#4CAF50', # Green/Aggressive
        'profile': 'Venture/Aggressive Growth Profile' # Simplified Name
    }
}


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
    
    # 1. Feature Selection & Target Definition - ADDING MORE FEATURES
    FEATURES = [
        'person_age', 'person_income', 'person_home_ownership', 
        'person_emp_length', 'loan_amnt', 'loan_int_rate', 
        'loan_percent_income', 'loan_status', # Existing
        'loan_grade', # NEW FEATURE
        'cb_person_default_on_file', # NEW FEATURE
        'cb_person_cred_hist_length' # NEW FEATURE
    ]
    
    # Keep only required columns and the target
    df = df[FEATURES + [TARGET_COLUMN]].copy()
    
    # 2. Basic Cleaning and Imputation
    df = df.dropna(subset=['person_age', 'person_income', TARGET_COLUMN])
    
    # Clean up and limit employee length to avoid outliers/noise (e.g., max 40 years)
    df['person_emp_length'] = df['person_emp_length'].clip(upper=40) 
    df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].clip(upper=40)
    
    # Limit rows to speed up demonstration
    df = df.sample(n=NUM_SAMPLES, random_state=42)
    
    return df, FEATURES

# --- 2. MODEL TRAINING AND CACHING ---
@st.cache_resource
def train_and_cache_model(df, features, target_column):
    
    X = df[features]
    y = df[target_column]

    # Define feature types based on the new dataset
    numerical_features = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'loan_status', # Binary, treated as numerical
        'cb_person_cred_hist_length' # NEW NUMERICAL FEATURE
    ]
    
    categorical_features = [
        'person_home_ownership', 
        'loan_grade', # NEW CATEGORICAL FEATURE
        'cb_person_default_on_file' # NEW CATEGORICAL FEATURE
    ]
    
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

    # Use a Classifier for multi-class prediction
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
loan_grade_options = sorted(data_df['loan_grade'].unique().tolist()) # Sort A, B, C...
default_on_file_options = data_df['cb_person_default_on_file'].unique().tolist()

# --- 3. APP CONFIGURATION AND STYLING ---

st.set_page_config(page_title="Robo Advisor for Personal Finance", page_icon="🏦", layout="wide")

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

st.title("💰 Robo Advisor for Personal Finance")
st.write("Get a personalized financial profile prediction and a tailored strategy based on your debt, income, and credit history.")

# --- 4. USER INPUT FORM (NOW WITH MORE FIELDS) ---

with st.form("intent_advisor_form", clear_on_submit=False):
    st.subheader("Input Your Comprehensive Financial Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years):", min_value=18, max_value=80, value=35, key='age')
        income = st.number_input("Annual Gross Income ($):", min_value=10000, value=85000, step=5000, key='income')
        emp_length = st.number_input("Employment History (Years):", min_value=0.0, max_value=40.0, value=5.0, step=0.5, key='emp_length')
    
    with col2:
        home_ownership = st.selectbox("Home Ownership Status:", home_options, index=1, key='home')
        loan_amnt = st.number_input("Total Debt Amount ($):", min_value=0, value=15000, step=1000, key='loan_amnt')
        loan_int_rate = st.number_input("Average Interest Rate on Debt (%):", min_value=0.0, max_value=30.0, value=12.5, step=0.1, format="%.1f", key='int_rate')

    with col3:
        # NEW INPUTS START HERE
        loan_grade = st.selectbox("Current Debt Grade (A=Best, G=Worst):", loan_grade_options, index=0, key='loan_grade')
        default_on_file = st.selectbox("Has Default Record on File?", default_on_file_options, index=1, key='default_on_file', format_func=lambda x: "Yes (Y)" if x == 'Y' else "No (N)")
        cred_hist_length = st.number_input("Credit History Length (Years):", min_value=0, max_value=40, value=10, key='cred_hist_length')
        # EXISTING INPUT
        loan_status = st.selectbox("Past Financial Stability (Proxy):", [0, 1], format_func=lambda x: "Stable (0)" if x == 0 else "Risky (1)", key='loan_status') 
        
        loan_percent_income = loan_amnt / income if income > 0 else 0
        st.metric(label="Calculated Debt-to-Income (DTI) Ratio:", 
                  value=f"{loan_percent_income:.2f}",
                  help="Total Debt Amount divided by Annual Income. A key factor in risk analysis.")
        
    st.markdown("---")
    submitted = st.form_submit_button("Get Personalized Strategy")


# --- 5. PREDICTION AND DISPLAY ---
if submitted:
    
    # Ensure all 11 features are included in the user_data DataFrame
    user_data = pd.DataFrame([{
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'loan_status': loan_status,
        'loan_grade': loan_grade, # NEW
        'cb_person_default_on_file': default_on_file, # NEW
        'cb_person_cred_hist_length': cred_hist_length # NEW
    }])
    
    # Ensure columns match the model's feature order
    user_data = user_data[model_features]

    with st.spinner("Analyzing profile and generating strategy..."):
        
        try:
            # Predict probability of belonging to each class (multi-class: 6 intents)
            probabilities = model.predict_proba(user_data)[0]
            
            # Get the predicted intent (class label)
            predicted_intent = model.classes_[np.argmax(probabilities)]
            
            # Get the advice map for the predicted intent
            advice_map = INTENT_ADVICE.get(predicted_intent, INTENT_ADVICE['PERSONAL'])
            
            st.markdown("---")
            
            st.subheader("✅ Personalized Financial Strategy Analysis")
            
            # Display Prediction
            col_pred, col_summary = st.columns([1, 2])
            
            with col_pred:
                st.markdown(f"""
                    <div class="container" style='border: 3px solid {advice_map['color']}; text-align:center;'>
                        <p style='font-size:18px; color: #333; font-weight: bold;'>Predicted Financial Profile</p>
                        <h1 style='font-size:32px; color: {advice_map['color']};'>{advice_map['profile']}</h1>
                        <p>(Strategy is tailored to this predicted profile)</p>
                    </div>
                """, unsafe_allow_html=True)

            with col_summary:
                st.success(f"**Recommended Strategy: {advice_map['title']}**")
                st.markdown(f"""
                    <div class="container" style='border-left: 5px solid {advice_map['color']};'>
                        <p style='font-size:16px;'>{advice_map['advice']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            st.subheader("📊 Key Metrics Snapshot")
            
            col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
            with col_metric_1:
                 st.metric(label="Debt-to-Income (DTI) Ratio", 
                          value=f"{loan_percent_income:.2f}")
            with col_metric_2:
                st.metric(label="Annual Debt Amount", 
                          value=f"${loan_amnt:,.0f}")
            with col_metric_3:
                 st.metric(label="Credit History Length", 
                          value=f"{cred_hist_length} Years")

            
            # --- Multi-Bar Confidence Graph ---
            st.subheader("Model Confidence Breakdown: All Predicted Profiles")
            
            # Map model classes (DEBTCONSOLIDATION, etc.) to simplified profile names
            profile_names = [INTENT_ADVICE[c]['profile'] for c in model.classes_]
            
            prob_df = pd.DataFrame({
                'Predicted Profile': profile_names,
                'Confidence': probabilities
            })
            
            # Sort for better visualization and ensure all intents are visible
            prob_df = prob_df.set_index('Predicted Profile').sort_values('Confidence', ascending=False)
            
            st.bar_chart(prob_df, y='Confidence')
            # --- END Multi-Bar Graph ---
            
            st.success("Analysis Complete!")
            
            # --- Balloons Animation ---
            st.balloons()
            # --- END Balloons Animation ---
            
        except Exception as e:
            st.error(f"Prediction error: Could not process input. Please ensure all number fields are valid. Detailed error: {e}")
            
