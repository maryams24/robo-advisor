import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

# --- 1. DATA LOADING AND PREPARATION (Using data.csv for budgeting) ---

FILE_NAME = 'data.csv' # Switched to the budgeting dataset
# TARGET: Predicts the user's Occupation (as a proxy for spending profile)
TARGET_COLUMN = 'Occupation' 
NUM_SAMPLES = 20000 

# Define advice mapped to the categories (Occupation/Profile)
PROFILE_ADVICE = {
    'Professional': {
        'title': "Targeted Savings & High-Yield Investment Strategy",
        'advice': "Your income is stable, but high discretionary spending (Eating Out/Entertainment) is likely eroding savings. **Reduce Eating Out by 15%** and investigate automated investing (e.g., 401k match, Roth IRA) to meet your savings goals.",
        'color': '#2196F3', # Blue/Growth
        'profile': 'Stable Professional Saver' 
    },
    'Self_Employed': {
        'title': "Variable Income Stabilization & Tax Saving Strategy",
        'advice': "Your income is variable. Your top priority is building a **larger cash reserve (12 months)** to smooth monthly fluctuations. Focus on reducing 'Miscellaneous' spending and set aside funds quarterly for taxes.",
        'color': '#FF9800', # Orange/Caution
        'profile': 'Entrepreneurial Saver Profile' 
    },
    'Student': {
        'title': "Essential Spending Optimization & Income Generation Strategy",
        'advice': "Your budget is tight. **Groceries and Rent are the biggest levers.** Explore cheaper alternatives for groceries (e.g., meal prepping) and consider a part-time income source to increase your 'Disposable Income' for savings.",
        'color': '#F44336', # Red/Urgent
        'profile': 'Optimized Student Saver' 
    },
    'Retired': {
        'title': "Fixed Income Preservation & Healthcare Strategy",
        'advice': "Your focus should be on **preserving capital and minimizing healthcare costs**. Review your Utilities for potential efficiency gains (e.g., energy audit). Ensure your withdrawal strategy minimizes tax liability.",
        'color': '#4CAF50', # Green/Preservation
        'profile': 'Conservative Retired Saver' 
    }
}


@st.cache_data(show_spinner=False)
def load_and_clean_data(file_name):
    """Loads, cleans, and prepares the Budgeting data for classification."""
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_name}' not found. Please upload it or ensure the name is correct.")
        st.stop()
        
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # 1. Feature Selection & Target Definition
    # Features for predicting spending profile
    FEATURES = [
        'income', 'age', 'dependents', 'city_tier', 
        'rent', 'groceries', 'transport', 'eating_out', 
        'entertainment', 'utilities', 'healthcare', 'education'
    ]
    
    # Keep only required columns and the target
    df = df[FEATURES + [TARGET_COLUMN]].copy()
    
    # Filter out categories not covered in advice map for cleaner model output
    df = df[df[TARGET_COLUMN].isin(PROFILE_ADVICE.keys())]
    
    # 2. Basic Cleaning and Imputation
    df = df.dropna()
    
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
        'income', 'age', 'dependents', 'rent', 'groceries', 
        'transport', 'eating_out', 'entertainment', 
        'utilities', 'healthcare', 'education'
    ]
    
    categorical_features = ['city_tier']
    
    # Create Preprocessing Pipelines
    numerical_pipeline = Pipeline(steps=[
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
city_options = data_df['city_tier'].unique().tolist()
occupation_options = data_df['Occupation'].unique().tolist()

# --- 3. APP CONFIGURATION AND STYLING ---

st.set_page_config(page_title="Robo Advisor for Savings and Budgeting", page_icon="📈", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #4CAF50; /* Green for Savings */
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {background-color: #45a049;}
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Robo Advisor for Savings and Budgeting")
st.write("Enter your monthly budget to get personalized advice on where to cut spending and maximize your savings rate.")

# --- 4. USER INPUT FORM (BUDGETING FIELDS) ---

with st.form("savings_advisor_form", clear_on_submit=False):
    st.subheader("Input Your Monthly Income and Spending")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.header("Income & Demographics")
        income = st.number_input("Annual Income ($):", min_value=10000, value=75000, step=5000, key='income')
        age = st.number_input("Age (Years):", min_value=18, max_value=80, value=30, key='age')
        dependents = st.number_input("Dependents:", min_value=0, max_value=10, value=0, key='dependents')
        city_tier = st.selectbox("City Cost of Living Tier:", city_options, index=1, key='city_tier')
    
    with col_r:
        st.header("Monthly Spending")
        # All inputs are converted to monthly estimates from annual data features
        rent = st.number_input("Rent/Mortgage ($/Month):", min_value=0, value=1500, step=100, key='rent')
        groceries = st.number_input("Groceries ($/Month):", min_value=0, value=500, step=50, key='groceries')
        transport = st.number_input("Transport ($/Month):", min_value=0, value=250, step=25, key='transport')
        eating_out = st.number_input("Eating Out ($/Month):", min_value=0, value=300, step=50, key='eating_out')
        entertainment = st.number_input("Entertainment ($/Month):", min_value=0, value=200, step=25, key='entertainment')
        utilities = st.number_input("Utilities ($/Month):", min_value=0, value=150, step=10, key='utilities')
        healthcare = st.number_input("Healthcare ($/Month):", min_value=0, value=100, step=10, key='healthcare')
        education = st.number_input("Education/Self-Improvement ($/Month):", min_value=0, value=50, step=10, key='education')
        
    st.markdown("---")
    submitted = st.form_submit_button("Get Personalized Savings Plan")


# --- 5. PREDICTION AND DISPLAY ---
if submitted:
    
    # Calculate approximate annual values for the model input
    # Note: The model was trained on annual data, so we convert monthly inputs back to annual estimates
    user_data = pd.DataFrame([{
        'income': income,
        'age': age,
        'dependents': dependents,
        'city_tier': city_tier,
        'rent': rent * 12,
        'groceries': groceries * 12,
        'transport': transport * 12,
        'eating_out': eating_out * 12,
        'entertainment': entertainment * 12,
        'utilities': utilities * 12,
        'healthcare': healthcare * 12,
        'education': education * 12
    }])
    
    # Ensure columns match the model's feature order
    user_data = user_data[model_features]

    with st.spinner("Analyzing spending profile and generating savings strategy..."):
        
        try:
            # Predict probability of belonging to each class (multi-class: 4 occupations/profiles)
            probabilities = model.predict_proba(user_data)[0]
            
            # Get the predicted profile (Occupation label)
            predicted_occupation = model.classes_[np.argmax(probabilities)]
            
            # Get the advice map for the predicted profile
            advice_map = PROFILE_ADVICE.get(predicted_occupation, PROFILE_ADVICE['Professional'])
            
            st.markdown("---")
            
            st.subheader("✅ Personalized Savings and Budget Analysis")
            
            # Display Prediction
            col_pred, col_summary = st.columns([1, 2])
            
            # Calculate Monthly Net Disposable Income (simplified, excluding taxes)
            total_monthly_spending = rent + groceries + transport + eating_out + entertainment + utilities + healthcare + education
            monthly_income = income / 12
            potential_monthly_savings = monthly_income - total_monthly_spending
            
            with col_pred:
                st.markdown(f"""
                    <div class="container" style='border: 3px solid {advice_map['color']}; text-align:center;'>
                        <p style='font-size:18px; color: #333; font-weight: bold;'>Predicted Financial Profile</p>
                        <h1 style='font-size:32px; color: {advice_map['color']};'>{advice_map['profile']}</h1>
                        <p>(Strategy tailored to this predicted spending profile)</p>
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
            
            st.subheader("📈 Your Savings Potential Snapshot")
            
            col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
            with col_metric_1:
                 st.metric(label="Estimated Monthly Savings (Pre-Tax)", 
                          value=f"${potential_monthly_savings:,.0f}")
            with col_metric_2:
                st.metric(label="Total Monthly Spending", 
                          value=f"${total_monthly_spending:,.0f}")
            with col_metric_3:
                 st.metric(label="Savings Rate (Estimated)", 
                          value=f"{((potential_monthly_savings / monthly_income) * 100):.1f}%")

            
            # --- Multi-Bar Confidence Graph ---
            st.subheader("Model Confidence Breakdown: All Predicted Profiles")
            
            # Map model classes (Occupation) to simplified profile names
            profile_names = [PROFILE_ADVICE[c]['profile'] for c in model.classes_]
            
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
            
