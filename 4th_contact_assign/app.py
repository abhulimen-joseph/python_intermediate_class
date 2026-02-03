import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏠 House Price Prediction Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Data preprocessing function
def preprocess_data(df):
    """
    Clean and preprocess the dataset
    """
    # Create a copy to avoid modifying original
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
            df_clean[f'{col}_year'] = df_clean[col].dt.year.astype(int)
            df_clean[f'{col}_month'] = df_clean[col].dt.month.astype(int)
            df_clean[f'{col}_day'] = df_clean[col].dt.day.astype(int)
            df_clean = df_clean.drop(col, axis=1)
    
    # Convert object columns to numeric where possible
    for col in df_clean.select_dtypes(include=['object']).columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Convert to regular Python float if it's a numpy float
            if df_clean[col].dtype in [np.float64, np.float32]:
                df_clean[col] = df_clean[col].astype(float)
            elif df_clean[col].dtype in [np.int64, np.int32]:
                df_clean[col] = df_clean[col].astype(int)
        except:
            # If conversion fails, use one-hot encoding for categorical columns with few unique values
            if df_clean[col].nunique() < 20:
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True).astype(int)
                df_clean = pd.concat([df_clean.drop(col, axis=1), dummies], axis=1)
            else:
                df_clean = df_clean.drop(col, axis=1)
    
    # Fill missing values with median for numeric columns
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Convert numpy types to Python native types
    for col in df_clean.columns:
        if df_clean[col].dtype == np.float64:
            df_clean[col] = df_clean[col].astype(float)
        elif df_clean[col].dtype == np.int64:
            df_clean[col] = df_clean[col].astype(int)
        elif df_clean[col].dtype == np.bool_:
            df_clean[col] = df_clean[col].astype(bool)
    
    return df_clean

# Helper function to make DataFrames Streamlit-safe
def make_streamlit_safe(df):
    """Convert DataFrame to be compatible with Streamlit"""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == np.float64:
            df_safe[col] = df_safe[col].astype(float)
        elif df_safe[col].dtype == np.int64:
            df_safe[col] = df_safe[col].astype(int)
        elif df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].astype(str)
    return df_safe

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/home--v1.png", width=100)
    st.title("Settings")
    
    # File upload option
    st.subheader("📁 Data Source")
    use_sample = st.checkbox("Use Sample Dataset", value=True)
    
    if not use_sample:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.warning("Please upload a CSV file or use the sample dataset")
            st.stop()
    else:
        # Use the provided file path
        try:
            df = pd.read_csv(r"C:\Users\HomePC\python_intermediate_class_2\4th_contact_assign\kc_house_data.csv")
            st.success(f"✅ Sample dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not load sample dataset: {str(e)}")
            st.info("Please upload your own CSV file below:")
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.stop()
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("⚙️ Model Parameters")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random State", 0, 100, 42)
    
    # Model selection
    st.subheader("🤖 Model Selection")
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
    if st.button("Run Analysis", type="primary"):
        st.session_state.run_analysis = True

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Main content
if 'df' in locals():
    # Display dataset information
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Dataset preview
    with st.expander("🔍 Dataset Preview & Details", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["First 10 Rows", "Last 10 Rows", "Data Types", "Statistics"])
        
        with tab1:
            st.dataframe(make_streamlit_safe(df.head(10)), width='stretch')
        
        with tab2:
            st.dataframe(make_streamlit_safe(df.tail(10)), width='stretch')
        
        with tab3:
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Missing Values': df.isnull().sum().values
            })
            st.dataframe(make_streamlit_safe(dtype_info), width='stretch')
        
        with tab4:
            st.dataframe(make_streamlit_safe(df.describe()), width='stretch')
    
    # Show columns with string data
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    if string_columns:
        with st.expander("String Columns Detected"):
            st.write("These columns contain non-numeric data and will be processed:")
            for col in string_columns:
                unique_vals = df[col].dropna().unique()[:5]
                st.write(f"**{col}**: {len(df[col].unique())} unique values. Sample: {list(unique_vals)}")
    
    # Run analysis when button is clicked
    if st.session_state.run_analysis:
        st.markdown("---")
        st.markdown("## Data Processing")
        
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
        
        # Split data
        st.info("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        st.success(f"✅ Data split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        st.markdown("---")
        st.markdown("## 📈 Model Training")
        
        # Define pipelines based on selection
        pipelines = {}
        if models_to_use["Linear Regression"]:
            pipelines["LinearRegression"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])
        
        if models_to_use["Random Forest"]:
            pipelines["RandomForest"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    random_state=random_state, 
                    n_estimators=100,
                    n_jobs=-1
                ))
            ])
        
        if models_to_use["Ridge Regression"]:
            pipelines["Ridge"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(random_state=random_state, alpha=1.0))
            ])
        
        if models_to_use["Lasso Regression"]:
            pipelines["Lasso"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(random_state=random_state, alpha=1.0))
            ])
        
        if models_to_use["Decision Tree"]:
            pipelines["DecisionTree"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", DecisionTreeRegressor(random_state=random_state, max_depth=10))
            ])
        
        if not pipelines:
            st.error("❌ Please select at least one model!")
            st.stop()
        
        # Train models and collect results
        results = []
        predictions_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model_pipeline) in enumerate(pipelines.items()):
            status_text.text(f"Training {name}... ({i+1}/{len(pipelines)})")
            
            try:
                # Train model
                model_pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model_pipeline.predict(X_train)
                y_test_pred = model_pipeline.predict(X_test)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model_pipeline, X_train, y_train, 
                    cv=cv_folds, scoring="r2", n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results.append({
                    "Model": name,
                    "Train MAE": train_mae,
                    "Test MAE": test_mae,
                    "Train RMSE": train_rmse,
                    "Test RMSE": test_rmse,
                    "Train R²": train_r2,
                    "Test R²": test_r2,
                    "CV R² Mean": cv_mean,
                    "CV R² Std": cv_std
                })
                
                # Store predictions for visualization
                predictions_data.append({
                    "Model": name,
                    "y_test": y_test.values,
                    "y_pred": y_test_pred,
                    "r2": test_r2
                })
                
                st.success(f"✅ {name} trained successfully")
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(pipelines))
        
        status_text.text("✅ All models trained successfully!")
        progress_bar.empty()
        
        if not results:
            st.error("❌ No models were successfully trained!")
            st.stop()
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Model Performance Comparison")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Format for display
        display_df = results_df.copy()
        for col in ['Train MAE', 'Test MAE', 'Train RMSE', 'Test RMSE']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        for col in ['Train R²', 'Test R²', 'CV R² Mean', 'CV R² Std']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(make_streamlit_safe(display_df), width='stretch')
        
        # Visualizations
        st.markdown("---")
        st.markdown("## 📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score Comparison
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars1 = ax1.bar(results_df["Model"], results_df["Test R²"], 
                color=['#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE'])
            ax1.set_title("Test R² Score by Model", fontsize=16, fontweight='bold')
            ax1.set_xlabel("Model", fontsize=12)
            ax1.set_ylabel("R² Score", fontsize=12)
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # MAE Comparison
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.bar(results_df["Model"], results_df["Test MAE"], 
                color=['#DC2626', '#EF4444', '#F87171', '#FCA5A5', '#FECACA'])
            ax2.set_title("Test MAE by Model", fontsize=16, fontweight='bold')
            ax2.set_xlabel("Model", fontsize=12)
            ax2.set_ylabel("MAE ($)", fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Best Model Analysis
        st.markdown("---")
        st.markdown("## 🏆 Best Model Analysis")
        
        if results_df.empty:
            st.warning("No results to display.")
        else:
            # Find best model based on Test R²
            best_idx = results_df["Test R²"].idxmax()
            best_model = results_df.iloc[best_idx]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Model", best_model["Model"])
            with col2:
                st.metric("Test R²", f"{best_model['Test R²']:.3f}")
            with col3:
                st.metric("Test MAE", f"${best_model['Test MAE']:,.2f}")
            with col4:
                st.metric("CV R² Mean", f"{best_model['CV R² Mean']:.3f}")
            
            # Actual vs Predicted for Best Model
            best_model_data = next((item for item in predictions_data if item["Model"] == best_model["Model"]), None)
            
            if best_model_data:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                
                # Scatter plot
                ax3.scatter(best_model_data["y_test"], best_model_data["y_pred"], 
                    alpha=0.6, s=30, color='#1E3A8A', label='Predictions')
                
                # Perfect prediction line
                max_val = max(best_model_data["y_test"].max(), best_model_data["y_pred"].max())
                min_val = min(best_model_data["y_test"].min(), best_model_data["y_pred"].min())
                ax3.plot([min_val, max_val], [min_val, max_val], 
                        color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
                
                ax3.set_title(f"Actual vs Predicted Prices - {best_model['Model']}", 
                    fontsize=16, fontweight='bold')
                ax3.set_xlabel("Actual Price ($)", fontsize=12)
                ax3.set_ylabel("Predicted Price ($)", fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Add R² text
                ax3.text(0.05, 0.95, f'R² = {best_model_data["r2"]:.3f}', 
                        transform=ax3.transAxes, fontsize=12,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig3)
        
        # Download results
        st.markdown("---")
        st.markdown("## 💾 Download Results")
        
        # Convert results to CSV
        csv = results_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("🔄 Run New Analysis"):
                st.session_state.run_analysis = False
                st.rerun()

else:
    # Initial state
    st.info("Please use the sidebar to load your dataset and configure the analysis.")
    st.markdown("""
    ### 📝 Instructions:
    1. **Load Data**: Upload your CSV or use the sample dataset
    2. **Configure Settings**: Adjust parameters in the sidebar
    3. **Select Models**: Choose which algorithms to train
    4. **Run Analysis**: Click the 'Run Analysis' button
    5. **View Results**: Explore metrics and visualizations
    """)