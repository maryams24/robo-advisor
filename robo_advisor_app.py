import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

FILE_NAME = 'data.csv' 
TARGET_COLUMN = 'Occupation' 
NUM_SAMPLES = 20000 

PROFILE_ADVICE = {
    'Professional': {
        'title': "Targeted Savings & High-Yield Investment Strategy",
        'advice': "Your income is stable, but high discretionary spending (Eating Out/Entertainment) is likely eroding savings. **Reduce Eating Out by 15%** and investigate automated investing (e.g., 401k match, Roth IRA) to meet your savings goals.",
        'color': '#2196F3',
        'profile': 'Stable Professional Saver' 
    },
    'Self_Employed': {
        'title': "Variable Income Stabilization & Tax Saving Strategy",
        'advice': "Your income is variable. Your top priority is building a **larger cash reserve (12 months)** to smooth monthly fluctuations. Focus on reducing 'Miscellaneous' spending and set aside funds quarterly for taxes.",
        'color': '#FF9800',
        'profile': 'Entrepreneurial Saver Profile' 
    },
    'Student': {
        'title': "Essential Spending Optimization & Income Generation Strategy",
        'advice': "Your budget is tight. **Groceries and Rent are the biggest levers.** Explore cheaper alternatives for groceries (e.g., meal prepping) and consider a part-time income source to increase your 'Disposable Income' for savings.",
        'color': '#F44336',
        'profile': 'Optimized Student Saver' 
    },
    'Retired': {
        'title': "Fixed Income Preservation & Healthcare Strategy",
        'advice': "Your focus should be on **preserving capital and minimizing healthcare costs**. Review your Utilities for potential efficiency gains (e.g., energy audit). Ensure your withdrawal strategy minimizes tax liability.",
        'color': '#4CAF50',
        'profile': 'Conservative Retired Saver' 
    }
}


@st.cache_data(show_spinner=False)
def load_and_clean_data(file_name):
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_name}' not found. Please upload it or ensure the name is correct.")
        st.stop()
        
    df.columns = df.columns.str.lower().str.strip()
    
    FEATURES = [
        'income', 'age', 'dependents', 'city_tier', 
        'rent', 'groceries', 'transport', 'eating_out', 
        'entertainment', 'utilities', 'healthcare', 'education'
    ]
    
    required_cols = FEATURES + [TARGET_COLUMN.lower()]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"""
            **Data Error: Missing Columns Detected!**
            
            The current model requires the budget/spending data from **`data.csv`**. 
            It appears you may be loading a different file, as these critical columns are missing:
            
            Missing Columns: **{', '.join(missing_cols)}**
            
            Please ensure the **`data.csv`** file is accessible to the code.
        """)
        st.stop()
    
    df = df[FEATURES + [TARGET_COLUMN.lower()]].copy()
    df.columns = FEATURES + [TARGET_COLUMN] 
    
    df = df[df[TARGET_COLUMN].str.strip().isin(PROFILE_ADVICE.keys())]
    
    df = df.dropna()
    
    if len(df) > NUM_SAMPLES:
        df = df.sample(n=NUM_SAMPLES, random_state=42)
    
    return df, FEATURES


@st.cache_resource
def train_and_cache_model(df, features, target_column):
    
    X = df[features]
    y = df[target_column]

    numerical_features = [
        'income', 'age', 'dependents', 'rent', 'groceries', 
        'transport', 'eating_out', 'entertainment', 
        'utilities', 'healthcare', 'education'
    ]
    
    categorical_features = ['city_tier']
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features), 
            ('cat', categorical_pipeline, categorical_features) 
        ],
        remainder='passthrough'
    )

    deep_learning_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25), 
        max_iter=500, 
        solver='adam', 
        random_state=42
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', deep_learning_model) 
    ])

    model.fit(X, y)

    return model, features

try:
    data_df, model_features = load_and_clean_data(FILE_NAME)
    model, model_features = train_and_cache_model(data_df, model_features, TARGET_COLUMN)
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()
    
city_options = data_df['city_tier'].unique().tolist()


st.set_page_config(page_title="Robo Advisor for Savings and Budgeting", page_icon="📈", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #4CAF50; 
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
        /* Custom styling for sliders/inputs */
        .stSlider label {
            font-weight: bold;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Robo Advisor for Savings and Budgeting (Deep Learning Powered)")
st.write("Enter your monthly budget to get personalized advice on where to cut spending and maximize your savings rate.")

with st.form("savings_advisor_form", clear_on_submit=False):
    st.subheader("Input Your Financial Data")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.header("👤 Income & Demographics")
        income = st.number_input("💰 Annual Income ($):", min_value=10000, value=75000, step=5000, key='income')
        # Using st.slider for better interaction
        age = st.slider("🎂 Age (Years):", min_value=18, max_value=80, value=30, key='age')
        dependents = st.slider("👨‍👩‍👧 Dependents:", min_value=0, max_value=10, value=0, key='dependents')
        city_tier = st.selectbox("🏙️ City Cost of Living Tier:", city_options, index=0 if len(city_options) > 0 else 0, key='city_tier')
    
    with col_r:
        st.header("🧾 Monthly Spending")
        rent = st.number_input("🏠 Rent/Mortgage ($/Month):", min_value=0, value=1500, step=100, key='rent')
        groceries = st.number_input("🛒 Groceries ($/Month):", min_value=0, value=500, step=50, key='groceries')
        transport = st.number_input("🚗 Transport ($/Month):", min_value=0, value=250, step=25, key='transport')
        eating_out = st.number_input("🍕 Eating Out ($/Month):", min_value=0, value=300, step=50, key='eating_out')
        entertainment = st.number_input("🎬 Entertainment ($/Month):", min_value=0, value=200, step=25, key='entertainment')
        utilities = st.number_input("💡 Utilities ($/Month):", min_value=0, value=150, step=10, key='utilities')
        healthcare = st.number_input("⚕️ Healthcare ($/Month):", min_value=0, value=100, step=10, key='healthcare')
        education = st.number_input("📚 Education/Self-Improvement ($/Month):", min_value=0, value=50, step=10, key='education')
        
    st.markdown("---")
    submitted = st.form_submit_button("Get Personalized Savings Plan")


if submitted:
    
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
    
    user_data = user_data[model_features]

    with st.spinner("Analyzing spending profile using Neural Network and generating savings strategy..."):
        
        try:
            probabilities = model.predict_proba(user_data)[0]
            
            predicted_occupation = model.classes_[np.argmax(probabilities)]
            
            advice_map = PROFILE_ADVICE.get(predicted_occupation, PROFILE_ADVICE['Professional'])
            
            st.markdown("---")
            
            st.subheader("✅ Personalized Savings and Budget Analysis")
            
            col_pred, col_summary = st.columns([1, 2])
            
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
                st.success(f"**Savings & Investment Action Plan: {advice_map['title']}**")
                st.markdown(f"""
                    <div class="container" style='border-left: 5px solid {advice_map['color']};'>
                        <p style='font-size:18px; font-weight: bold; color: #333;'>Your Personalized Financial Directive:</p>
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

            
            st.subheader("Model Confidence Breakdown: All Predicted Profiles")
            
            profile_names = [PROFILE_ADVICE[c]['profile'] for c in model.classes_]
            
            prob_df = pd.DataFrame({
                'Predicted Profile': profile_names,
                'Confidence': probabilities
            })
            
            prob_df = prob_df.set_index('Predicted Profile').sort_values('Confidence', ascending=False)
            
            st.bar_chart(prob_df, y='Confidence')
            
            st.success("Analysis Complete!")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction error: Could not process input. Please ensure all number fields are valid. Detailed error: {e}")
            
