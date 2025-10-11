import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib

# --- 1. Page Configuration, Styling ---

st.set_page_config(page_title="Robo-Advisor: Financial Profile Analysis", page_icon="📈", layout="centered")

# Custom CSS for a clean look
st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #1a73e8; /* Google Blue */
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
        .stTitle {color: #1a73e8; text-align: center;}
        .stSubheader {color: #3367d6;}
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

# --- Contextual Advice Mapping (New Feature) ---
FINANCIAL_ADVICE = {
    'Professional': {
        'title': "Growth-Focused Investment Strategy",
        'advice': "As a **Professional**, you likely have a stable income and a longer time horizon. Focus on a balanced portfolio with a strong emphasis on **equity** and **growth mutual funds** (70-80%). Maintain an emergency fund of 6 months' expenses, and increase your contribution to retirement accounts.",
        'color': 'green'
    },
    'Self_Employed': {
        'title': "Income Volatility & Protection Strategy",
        'advice': "As **Self-Employed**, income can be variable. Your priority should be a larger **emergency fund (8-12 months)**. Allocate a portion to low-risk, liquid assets like **Fixed Deposits** or **short-term bonds** (30-40%) before investing in higher-risk assets like stocks. Consider self-insurance for income protection.",
        'color': 'orange'
    },
    'Student': {
        'title': "Long-Term, High-Growth Strategy",
        'advice': "As a **Student**, your biggest asset is time. Focus on **compound interest**! Even small, consistent investments in low-cost **index funds** or **ETFs** (80-90%) will yield massive returns over decades. Limit debt and focus on skill-building investments (education).",
        'color': 'blue'
    },
    'Retired': {
        'title': "Capital Preservation & Income Strategy",
        'advice': "As **Retired**, capital preservation and stable income are key. Prioritize low-volatility assets like **Government Bonds**, **Fixed Deposits**, and **Debt Mutual Funds** (60-70%). Withdraw from your portfolio safely using the 4% rule. Avoid high-risk, speculative investments.",
        'color': 'red'
    },
    'Other': {
        'title': "Conservative and Balanced Strategy",
        'advice': "Your profile suggests a balanced approach. Maintain a mix of assets: **50% growth (stocks/mutual funds)** and **50% stability (bonds/deposits)**. Ensure all high-interest debt is cleared before increasing investments.",
        'color': 'gray'
    }
}


# --- 2. Model Training and Caching ---

@st.cache_resource
def train_and_cache_model():
    """
    Trains the machine learning model to predict 'Occupation' and caches it.
    """
    FILE_NAME = 'data.csv'
    TARGET_COLUMN = 'Occupation'
    
    # Define features based on the available data columns
    FEATURES = [
        'Income', 'Age', 'Dependents', 'City_Tier', 'Rent', 'Loan_Repayment', 
        'Desired_Savings_Percentage', 'Disposable_Income'
    ]

    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        st.error(f"Error: Data file '{FILE_NAME}' not found. Please ensure it is in the same directory.")
        st.stop()
    
    # 1. Column Cleaning
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
    # 2. Select Features and Target
    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    # 3. Data Cleaning and Preprocessing
    categorical_features = ['City_Tier']
    numerical_features = [col for col in FEATURES if col not in categorical_features]
    
    # Convert numerical columns to numeric, coercing errors to NaN
    for col in numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop rows with any missing values in features or target
    full_data = pd.concat([X, y], axis=1).dropna(subset=FEATURES + [TARGET_COLUMN])
    X = full_data[FEATURES]
    y = full_data[TARGET_COLUMN]
    
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features), # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-hot encode categorical features
        ],
        remainder='passthrough'
    )

    # Create and train a machine learning pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)

    # Return the trained model, the features, and the cleaned dataframe
    return model, FEATURES, full_data

# Train and load the model at the start of the app
model, model_features, full_data = train_and_cache_model()

# --- 3. User Input Form ---

with st.form("advisor_form", clear_on_submit=False):
    st.subheader("Tell us about your Financial Profile")

    # The form now only collects the features used by the adapted model
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age:", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Number of Dependents:", min_value=0, max_value=10, value=1)
        city_tier = st.selectbox("City Tier:", sorted(full_data['City_Tier'].unique()))
        income = st.number_input("Annual Income:", min_value=0.0, value=50000.0, format="%.2f")
        
    with col2:
        rent = st.number_input("Monthly Rent/Housing Payment:", min_value=0.0, value=1000.0, format="%.2f")
        loan_repayment = st.number_input("Monthly Loan Repayment:", min_value=0.0, value=500.0, format="%.2f")
        desired_savings_percentage = st.slider("Desired Monthly Savings Percentage:", min_value=0.0, max_value=50.0, value=15.0, step=0.5, format="%.1f%%")
        # Calculate a proxy for Disposable_Income: Income - Rent - Loan_Repayment
        disposable_income = income - (rent * 12) - (loan_repayment * 12)
        st.metric("Estimated Annual Disposable Income (For Model):", f"$$ {disposable_income:,.2f}")
        
    submitted = st.form_submit_button("Get Personalized Strategy")

# --- 4. Prediction and Display ---

if submitted:
    # Create a DataFrame from user inputs
    user_data = pd.DataFrame([{
        'Income': income,
        'Age': age,
        'Dependents': dependents,
        'City_Tier': city_tier,
        'Rent': rent,
        'Loan_Repayment': loan_repayment,
        'Desired_Savings_Percentage': desired_savings_percentage,
        'Disposable_Income': disposable_income # Use the calculated proxy
    }])

    # Ensure the columns are in the same order as the model's training data
    user_data = user_data[model_features]

    # Make a prediction
    try:
        # Predict the most likely Occupation class
        prediction = model.predict(user_data)
        predicted_occupation = prediction[0]
        
        # Get the contextual advice
        advice_map = FINANCIAL_ADVICE.get(predicted_occupation, FINANCIAL_ADVICE['Other'])

        st.subheader("Your Predicted Financial Profile & Strategy")
        st.markdown(f"**We've analyzed your inputs and predict your profile aligns closest with a:** <span style='color:green; font-weight:bold;'>{predicted_occupation}</span>", unsafe_allow_html=True)
        st.success(f"Recommended Strategy: {advice_map['title']}")
        
        # Display the main advice
        st.markdown(f"""
        <div class="container" style='border: 2px solid #{advice_map['color']};'>
            <p style='font-size:16px;'>{advice_map['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show the probability of each class
        probabilities = model.predict_proba(user_data)[0]
        class_labels = model.classes_
        prob_df = pd.DataFrame({'Predicted Occupation': class_labels, 'Confidence': probabilities})
        prob_df = prob_df.sort_values('Confidence', ascending=False).reset_index(drop=True)
        prob_df['Confidence'] = (prob_df['Confidence'] * 100).round(2).astype(str) + '%'
        
        st.info("Confidence Score for Predicted Profiles:")
        st.table(prob_df)

        # Add a bar chart for visual clarity
        st.subheader("Confidence Scores Visualized")
        # Convert confidence back to float for the chart
        chart_df = prob_df.copy()
        chart_df['Confidence'] = chart_df['Confidence'].str.replace('%', '').astype(float) / 100
        st.bar_chart(chart_df, x='Predicted Occupation', y='Confidence')
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.warning("Please check your input values and try again.")
        
    st.balloons()