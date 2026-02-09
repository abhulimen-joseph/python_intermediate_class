import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Css
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 0.5rem 0;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏠 House Price Prediction Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

st.markdown('<h1 class="main-header">🏠 House Price Prediction Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Data preprocessing function
def preprocess_data(df):
    # Create a copy 
    df_clean = df.copy()
    
    # Drop columns that are completely null
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Convert date columns (like '20140527T000000') to datetime
    date_columns = []
    for col in df_clean.columns:
        # Check if column contains date-like strings
        sample = df_clean[col].dropna().iloc[0] if len(df_clean[col].dropna()) > 0 else ''
        if isinstance(sample, str) and 'T' in sample and len(sample) >= 8:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                date_columns.append(col)
            except:
                pass
    
    # Extract features from date columns
    for col in date_columns:
        if df_clean[col].dtype == 'datetime64[ns]':
            df_clean[f'{col}_year'] = df_clean[col].dt.year
            df_clean[f'{col}_month'] = df_clean[col].dt.month
            df_clean[f'{col}_day'] = df_clean[col].dt.day
            df_clean = df_clean.drop(col, axis=1)
    
    # Convert object columns to numeric where possible
    for col in df_clean.select_dtypes(include=['object']).columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        except:
            # If conversion fails, use one-hot encoding for categorical columns with few unique values
            if df_clean[col].nunique() < 20:
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                df_clean = pd.concat([df_clean.drop(col, axis=1), dummies], axis=1)
            else:
                df_clean = df_clean.drop(col, axis=1)
    
    # Fill missing values with median for numeric columns
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/home--v1.png", width=100)
    st.title("Settings")
    
    # File upload option
    st.subheader(" Data Source")
    use_sample = st.checkbox("Use Sample Dataset", value=True)
    
    if not use_sample:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or use the sample dataset")
            st.stop()
    else:
        # Use the provided file path
        try:
            df = pd.read_csv(r"C:\Users\HomePC\python_intermediate_class_2\4th_contact_assign\kc_house_data.csv")
            st.success("✅ Sample dataset loaded successfully!")
        except:
            st.error("Could not load sample dataset. Please upload your own file.")
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.stop()
    
    st.markdown("---")

    # Model parameters
    st.subheader("Model Parameters")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random State", 0, 100, 42)

    # Model selection
    st.subheader("Model Selection")
    models_to_use = {
        "Linear Regression": st.checkbox("Linear Regression", True),
        "Random Forest": st.checkbox("Random Forest", True),
        "Ridge Regression": st.checkbox("Ridge Regression", True),
        "Lasso Regression": st.checkbox("Lasso Regression", True),
        "Decision Tree": st.checkbox("Decision Tree", True)
    }

    # Cross-validation folds
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    st.markdown("---")
    if st.button("Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if 'df' in locals():
    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Dataset Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Dataset preview
    with st.expander(" Dataset Preview", expanded=True):
        tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Statistics"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df.tail(10), use_container_width=True)
        
        with tab3:
            st.dataframe(df.describe(), use_container_width=True)
    
    # Data Quality Check
    with st.expander("Data Quality Check"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Null values check
            null_counts = df.isnull().sum()
            null_df = pd.DataFrame({
                'Column': null_counts.index,
                'Null Count': null_counts.values,
                'Percentage': (null_counts.values / len(df)) * 100
            })
            null_df = null_df[null_df['Null Count'] > 0]
            
            if len(null_df) > 0:
                st.warning(f"⚠️ Found {len(null_df)} columns with missing values")
                st.dataframe(null_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            # Data types
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)

     # Run analysis when button is clicked
    if st.session_state.run_analysis:
        st.markdown("---")
        st.markdown("## Data preprocessing")
    
        with st.spinner("Preprocessing data..."):
            # Clean the data
            df_clean = preprocess_data(df)
            
            # Check if 'price' column exists
            if 'price' not in df_clean.columns:
                st.error("'price' column not found in the dataset after preprocessing!")
                st.write("Available columns:", df_clean.columns.tolist())
                st.stop()

            # Separate features and target
            X = df_clean.drop("price", axis=1)
            y = df_clean["price"]

            # Show preprocessing summary
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Data preprocessing complete!")
                st.write(f"Original shape: {df.shape}")
                st.write(f"Processed shape: {df_clean.shape}")
                st.write(f"Features: {X.shape[1]}")
            
            with col2:
                st.info("Processed data types:")
                dtype_counts = pd.Series(X.dtypes).value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} columns")
    

