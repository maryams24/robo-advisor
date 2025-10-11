import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- LOTTIE ANIMATION HELPERS ---

def load_lottieurl(url: str):
    """Fetches Lottie JSON data from a URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        # Handle connection errors gracefully without crashing the app
        return None

@st.cache_resource
def load_lottie_animations():
    """Caches both the loading and success Lottie animations."""
    
    # 1. Loading/Analysis Animation (used while calculating)
    LOTTIE_ANALYSIS_URL = "https://lottie.host/17498c19-75a0-43a7-89b3-ec3199859a0f/94178xN69F.json"
    analysis_animation = load_lottieurl(LOTTIE_ANALYSIS_URL)

    # 2. Success/Checkmark Animation (used after results are displayed)
    # A professional checkmark animation
    LOTTIE_SUCCESS_URL = "https://lottie.host/6f02a3a5-1d0e-4340-9b4e-862d6d03d4c8/Gf03u69YF8.json"
    success_animation = load_lottieurl(LOTTIE_SUCCESS_URL)

    return analysis_animation, success_animation

lottie_analysis, lottie_success = load_lottie_animations()

# --- 1. APP CONFIGURATION AND STYLING 🎨 ---

st.set_page_config(page_title="Robo-Advisor: Financial Profile Analysis", page_icon="📈", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {background-color: #1558b3;}
        .stTextInput, .stSelectbox, .stNumberInput {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI-Powered Financial Profile Advisor")
st.write("Get personalized financial strategy advice based on your profile.")


# --- 2. CONTEXTUAL ADVICE MAPPING (Strategy Database) ---
FINANCIAL_ADVICE = {
    'Professional': {
        'title': "Growth-Focused Investment Strategy (Balanced to Aggressive)",
        'advice': "As a **Professional**, you likely have a stable income and a longer time horizon. Focus on a balanced portfolio with a strong emphasis on **equity** and **growth mutual funds** (70-80%). Maintain an emergency fund of 6 months' expenses, and increase your contribution to retirement accounts.",
        'color': '#4CAF50'
    },
    'Self_Employed': {
        'title': "Income Volatility & Protection Strategy (Conservative)",
        'advice': "As **Self-Employed**, income can be variable. Your priority should be a larger **emergency fund (8-12 months)**. Allocate a portion to low-risk, liquid assets like **Fixed Deposits** or **short-term bonds** (30-40%) before investing in higher-risk assets like stocks. Consider self-insurance for income protection.",
        'color': '#FF9800'
    },
    'Student': {
        'title': "Long-Term, High-Growth Strategy (Aggressive)",
        'advice': "As a **Student**, your biggest asset is time. Focus on **compound interest**! Even small, consistent investments in low-cost **index funds** or **ETFs** (80-90%) will yield massive returns over decades. Limit debt and focus on skill-building investments (education).",
        'color': '#2196F3'
    },
    'Retired': {
        'title': "Capital Preservation & Income Strategy (Very Conservative)",
        'advice': "As **Retired**, capital preservation and stable income are key. Prioritize low-volatility assets like **Government Bonds**, **Fixed Deposits**, and **Debt Mutual Funds** (60-70%). Withdraw from your portfolio safely using the 4% rule. Avoid high-risk, speculative investments.",
        'color': '#F44336'
    },
    'Other': {
        'title': "Conservative and Balanced Strategy (Default)",
        'advice': "Your profile suggests a balanced approach. Maintain a mix of assets: **50% growth (stocks/mutual funds)** and **50% stability (bonds/deposits)**. Ensure all high-interest debt is cleared before increasing investments.",
        'color': '#9E9E9E'
    }
}


# --- 3. MODEL TRAINING AND CACHING 🧠 ---

@st.cache_resource
def train_and_cache_model():
    FILE_NAME = 'data.csv'
    TARGET_COLUMN = 'Occupation'
    
    FEATURES = [
        'Income', 'Age', 'Dependents', 'City_Tier', 'Rent', 'Loan_Repayment', 
        'Desired_Savings_Percentage', 'Disposable_Income'
    ]

    try:
        # Assuming 'data.csv' is accessible in the environment
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        st.error(f"Error: Data file '{FILE_NAME}' not found. Please ensure it is in the same directory.")
        st.stop()
    
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    categorical_features = ['City_Tier']
    numerical_features = [col for col in FEATURES if col not in categorical_features]
    
    full_data = pd.concat([X, y], axis=1).dropna(subset=FEATURES + [TARGET_COLUMN])
    X = full_data[FEATURES]
    y = full_data[TARGET_COLUMN]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)

    return model, FEATURES, full_data

model, model_features, full_data = train_and_cache_model()


# --- 4. USER INPUT FORM 📋 ---

with st.form("advisor_form", clear_on_submit=False):
    st.subheader("Tell us about your Financial Profile")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age:", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Number of Dependents:", min_value=0, max_value=10, value=1)
        city_tier = st.selectbox("City Tier:", sorted(full_data['City_Tier'].unique()))
        income = st.number_input("Annual Income:", min_value=0.0, value=50000.0, format="%.2f", help="Your total pre-tax annual income.")
        
    with col2:
        rent = st.number_input("Monthly Rent/Housing Payment:", min_value=0.0, value=1000.0, format="%.2f")
        loan_repayment = st.number_input("Monthly Loan Repayment:", min_value=0.0, value=500.0, format="%.2f")
        desired_savings_percentage = st.slider("Desired Monthly Savings Percentage:", min_value=0.0, max_value=50.0, value=15.0, step=0.5, format="%.1f%%")
        
        disposable_income = income - (rent * 12) - (loan_repayment * 12)
        
        # FIX: Using single '$' for currency display
        st.metric("Estimated Annual Disposable Income:", f"$ {disposable_income:,.2f}", help="This is used by the model for classification.")
        
    submitted = st.form_submit_button("Get Personalized Strategy")


# --- 5. PREDICTION AND DISPLAY 📊 ---

if submitted:
    
    # 5a. Show Lottie analysis animation during calculation
    if lottie_analysis:
        # Using columns to center the animation and make it prominent
        anim_col1, anim_col2, anim_col3 = st.columns([1, 2, 1])
        with anim_col2:
            st_lottie(lottie_analysis, height=200, key="analysis_animation", speed=1, loop=True)
    
    with st.spinner("Analyzing your financial profile and generating strategy..."):
        
        user_data = pd.DataFrame([{
            'Income': income,
            'Age': age,
            'Dependents': dependents,
            'City_Tier': city_tier,
            'Rent': rent,
            'Loan_Repayment': loan_repayment,
            'Desired_Savings_Percentage': desired_savings_percentage,
            'Disposable_Income': disposable_income 
        }])

        user_data = user_data[model_features]

        try:
            prediction = model.predict(user_data)
            predicted_occupation = prediction[0]
            
            advice_map = FINANCIAL_ADVICE.get(predicted_occupation, FINANCIAL_ADVICE['Other'])

            # Add a clear separator or markdown to indicate the transition from analysis to results
            st.markdown("---")
            
            st.subheader("✅ Strategy Generation Complete!")
            st.markdown(f"**We've analyzed your inputs and predict your profile aligns closest with a:** <span style='color:#1a73e8; font-weight:bold;'>{predicted_occupation}</span>", unsafe_allow_html=True)
            st.success(f"**Recommended Strategy:** {advice_map['title']}")
            
            st.markdown(f"""
            <div class="container" style='border: 2px solid {advice_map['color']};'>
                <p style='font-size:16px;'>{advice_map['advice']}</p>
            </div>
            """, unsafe_allow_html=True)

            probabilities = model.predict_proba(user_data)[0]
            class_labels = model.classes_
            prob_df = pd.DataFrame({'Predicted Occupation': class_labels, 'Confidence': probabilities})
            prob_df = prob_df.sort_values('Confidence', ascending=False).reset_index(drop=True)
            
            prob_df_display = prob_df.copy()
            prob_df_display['Confidence'] = (prob_df_display['Confidence'] * 100).round(2).astype(str) + '%'
            
            st.info("Confidence Score for Predicted Profiles:")
            st.table(prob_df_display)

            st.subheader("Confidence Scores Visualized")
            st.bar_chart(prob_df, x='Predicted Occupation', y='Confidence')
            
            # CELEBRATION: Show Lottie Success/Checkmark animation after the results are fully displayed
            if lottie_success:
                st_lottie(lottie_success, height=100, key="final_success_animation", loop=False, speed=1)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.warning("Please check your input values and try again.")
        
