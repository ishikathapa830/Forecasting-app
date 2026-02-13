import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False

# Try importing Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# Try importing LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# Page config
st.set_page_config(page_title="Sales Forecasting", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("📊 Advanced Sales Forecasting Application")
st.markdown("""
Predict future sales using advanced time-series models (Prophet, ARIMA, LSTM).
Analyze trends, seasonality, and promotional impacts on sales.
""")

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# Sidebar - File Upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    with st.sidebar.expander("📋 Data Info", expanded=False):
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        st.write("Columns:", df.columns.tolist())
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Data Exploration", "⚙️ Model Configuration", "🎯 Forecasting & Results"])
    
    # TAB 1: Data Exploration
    with tab1:
        st.header("Data Exploration & Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape}")
        
        with col2:
            st.subheader("Data Statistics")
            st.dataframe(df.describe())
        
        st.subheader("Data Preprocessing")
        
        col_date, col_sales = st.columns(2)
        
        with col_date:
            date_col = st.selectbox("Select Date Column", df.columns, key="date_col")
        
        with col_sales:
            sales_col = st.selectbox("Select Sales Column", df.columns, key="sales_col")
        
        # Preprocessing options
        col_promo, col_outlier = st.columns(2)
        
        with col_promo:
            has_promo = st.checkbox("Data has Promotional column?", False)
            promo_col = None
            if has_promo:
                promo_col = st.selectbox("Select Promotional Column", 
                                        [col for col in df.columns if col not in [date_col, sales_col]])
        
        with col_outlier:
            remove_outliers = st.checkbox("Remove Outliers (IQR method)?", True)
        
        # Process data
        if st.button("🔄 Process Data", key="process_btn"):
            try:
                df_processed = df.copy()
                
                # Convert date
                df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
                df_processed = df_processed.dropna(subset=[date_col])
                
                # Convert sales to numeric
                df_processed[sales_col] = pd.to_numeric(df_processed[sales_col], errors='coerce')
                df_processed = df_processed.dropna(subset=[sales_col])
                
                # Sort by date
                df_processed = df_processed.sort_values(date_col)
                
                # Handle missing values
                df_processed[sales_col] = df_processed[sales_col].interpolate(method='linear')
                
                # Remove outliers
                if remove_outliers:
                    Q1 = df_processed[sales_col].quantile(0.25)
                    Q3 = df_processed[sales_col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_processed = df_processed[
                        (df_processed[sales_col] >= Q1 - 1.5*IQR) & 
                        (df_processed[sales_col] <= Q3 + 1.5*IQR)
                    ]
                
                # Prepare for Prophet/LSTM (rename columns)
                df_processed_model = df_processed[[date_col, sales_col]].copy()
                df_processed_model.columns = ['ds', 'y']
                
                if has_promo and promo_col:
                    df_processed_model['promo'] = df_processed[promo_col].values
                
                st.session_state.df_processed = df_processed_model
                st.session_state.has_promo = has_promo
                st.session_state.promo_col = promo_col
                st.session_state.original_col_names = (date_col, sales_col)
                
                st.success("✅ Data processed successfully!")
                st.write(f"**Processed rows:** {len(df_processed_model)}")
                st.dataframe(df_processed_model.head(10))
                
                # Visualization
                fig = px.line(df_processed_model, x='ds', y='y', 
                             title='Sales Over Time', markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
    
    # TAB 2: Model Configuration
    with tab2:
        st.header("Model Configuration")
        
        if st.session_state.df_processed is None:
            st.warning("⚠️ Please process data in the Data Exploration tab first.")
        else:
            df_model = st.session_state.df_processed
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_days = st.number_input("Forecast Days", min_value=1, max_value=365, value=30)
            
            with col2:
                model_type = st.selectbox("Select Model", 
                                         ["Prophet", "ARIMA", "LSTM"] if all([PROPHET_AVAILABLE, STATSMODELS_AVAILABLE, LSTM_AVAILABLE])
                                         else ["Prophet"] if PROPHET_AVAILABLE
                                         else ["ARIMA"] if STATSMODELS_AVAILABLE
                                         else ["LSTM"])
            
            with col3:
                test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20)
            
            st.session_state.forecast_days = forecast_days
            st.session_state.model_type = model_type
            st.session_state.test_size = test_size
            
            # Model-specific configurations
            st.subheader("Model-Specific Parameters")
            
            if model_type == "Prophet" and PROPHET_AVAILABLE:
                col1, col2, col3 = st.columns(3)
                with col1:
                    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
                with col2:
                    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
                with col3:
                    daily_seasonality = st.checkbox("Daily Seasonality", value=False)
                
                changepoint_prior = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
                seasonality_prior = st.slider("Seasonality Prior Scale", 1, 20, 10)
                
                st.session_state.prophet_yearly = yearly_seasonality
                st.session_state.prophet_weekly = weekly_seasonality
                st.session_state.prophet_daily = daily_seasonality
                st.session_state.prophet_changepoint = changepoint_prior
                st.session_state.prophet_seasonality = seasonality_prior
            
            elif model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.slider("AR Order (p)", 0, 5, 1)
                with col2:
                    d = st.slider("Differencing (d)", 0, 2, 1)
                with col3:
                    q = st.slider("MA Order (q)", 0, 5, 1)
                
                use_seasonal = st.checkbox("Use SARIMA (Seasonal ARIMA)", False)
                
                if use_seasonal:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        P = st.slider("Seasonal AR (P)", 0, 2, 1)
                    with col2:
                        D = st.slider("Seasonal Diff (D)", 0, 1, 1)
                    with col3:
                        Q = st.slider("Seasonal MA (Q)", 0, 2, 1)
                    with col4:
                        m = st.slider("Period (m)", 7, 365, 7)
                    
                    st.session_state.arima_order = (p, d, q)
                    st.session_state.arima_seasonal = (P, D, Q, m)
                else:
                    st.session_state.arima_order = (p, d, q)
                    st.session_state.arima_seasonal = None
            
            elif model_type == "LSTM" and LSTM_AVAILABLE:
                col1, col2, col3 = st.columns(3)
                with col1:
                    epochs = st.slider("Epochs", 10, 200, 50)
                with col2:
                    batch_size = st.slider("Batch Size", 16, 128, 32)
                with col3:
                    look_back = st.slider("Look-back Window", 1, 60, 10)
                
                st.session_state.lstm_epochs = epochs
                st.session_state.lstm_batch = batch_size
                st.session_state.lstm_lookback = look_back
    
    # TAB 3: Forecasting & Results
    with tab3:
        st.header("Forecasting & Results")
        
        if st.session_state.df_processed is None:
            st.warning("⚠️ Please process data in the Data Exploration tab first.")
        elif st.button("🚀 Generate Forecast", key="forecast_btn"):
            df_model = st.session_state.df_processed
            forecast_days = st.session_state.get('forecast_days', 30)
            model_type = st.session_state.get('model_type', 'Prophet')
            test_size = st.session_state.get('test_size', 20) / 100
            
            with st.spinner("🔄 Training model and generating forecast..."):
                try:
                    # Split data
                    split_idx = int(len(df_model) * (1 - test_size))
                    df_train = df_model.iloc[:split_idx].copy()
                    df_test = df_model.iloc[split_idx:].copy()
                    
                    if model_type == "Prophet" and PROPHET_AVAILABLE:
                        # Prophet model
                        model = Prophet(
                            yearly_seasonality=st.session_state.get('prophet_yearly', True),
                            weekly_seasonality=st.session_state.get('prophet_weekly', True),
                            daily_seasonality=st.session_state.get('prophet_daily', False),
                            changepoint_prior_scale=st.session_state.get('prophet_changepoint', 0.05),
                            seasonality_prior_scale=st.session_state.get('prophet_seasonality', 10),
                            seasonality_mode='additive'
                        )
                        
                        if 'promo' in df_train.columns:
                            model.add_regressor('promo')
                        
                        model.fit(df_train)
                        
                        # Make future dataframe
                        future = model.make_future_dataframe(periods=forecast_days)
                        if 'promo' in df_model.columns:
                            future['promo'] = 0
                        
                        forecast = model.predict(future)
                        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        
                    elif model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                        # ARIMA model
                        order = st.session_state.get('arima_order', (1, 1, 1))
                        seasonal_order = st.session_state.get('arima_seasonal', None)
                        
                        if seasonal_order:
                            model = SARIMAX(df_train['y'], order=order, seasonal_order=seasonal_order)
                        else:
                            model = ARIMA(df_train['y'], order=order)
                        
                        fitted_model = model.fit()
                        forecast_values = fitted_model.forecast(steps=len(df_test) + forecast_days)
                        
                        # Build forecast dataframe
                        last_date = df_model['ds'].iloc[-1]
                        future_dates = pd.date_range(start=df_model['ds'].iloc[-1] + timedelta(days=1), 
                                                    periods=forecast_days, freq='D')
                        
                        forecast_output = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': forecast_values[-forecast_days:],
                            'yhat_lower': forecast_values[-forecast_days:] - np.std(fitted_model.resid) * 1.96,
                            'yhat_upper': forecast_values[-forecast_days:] + np.std(fitted_model.resid) * 1.96
                        })
                    
                    elif model_type == "LSTM" and LSTM_AVAILABLE:
                        # LSTM model
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(df_train[['y']])
                        
                        look_back = st.session_state.get('lstm_lookback', 10)
                        
                        # Create sequences
                        X_train, y_train = [], []
                        for i in range(len(scaled_data) - look_back):
                            X_train.append(scaled_data[i:i+look_back])
                            y_train.append(scaled_data[i+look_back])
                        
                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        
                        # Build LSTM model
                        lstm_model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                            Dropout(0.2),
                            LSTM(50),
                            Dropout(0.2),
                            Dense(25),
                            Dense(1)
                        ])
                        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                        lstm_model.fit(X_train, y_train, 
                                     epochs=st.session_state.get('lstm_epochs', 50),
                                     batch_size=st.session_state.get('lstm_batch', 32),
                                     verbose=0)
                        
                        # Forecast
                        last_sequence = scaled_data[-look_back:]
                        predictions = []
                        
                        for _ in range(forecast_days):
                            pred = lstm_model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
                            predictions.append(pred[0][0])
                            last_sequence = np.append(last_sequence[1:], pred)
                        
                        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                        
                        future_dates = pd.date_range(start=df_model['ds'].iloc[-1] + timedelta(days=1), 
                                                    periods=forecast_days, freq='D')
                        
                        forecast_output = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': predictions,
                            'yhat_lower': predictions - np.std(predictions) * 0.1,
                            'yhat_upper': predictions + np.std(predictions) * 0.1
                        })
                    
                    st.session_state.forecast_results = {
                        'forecast': forecast_output,
                        'train': df_train,
                        'test': df_test
                    }
                    st.success("✅ Forecast generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during forecasting: {str(e)}")
        
        # Display results
        if st.session_state.forecast_results is not None:
            results = st.session_state.forecast_results
            forecast = results['forecast']
            df_train = results['train']
            df_test = results['test']
            
            st.subheader("📊 Forecast Visualization")
            
            # Combine data
            combined = pd.concat([
                df_train[['ds', 'y']].rename(columns={'y': 'Sales'}),
                df_test[['ds', 'y']].rename(columns={'y': 'Sales'})
            ])
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=combined['ds'], y=combined['Sales'],
                name='Historical Sales',
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
            # Test data
            fig.add_trace(go.Scatter(
                x=df_test['ds'], y=df_test['y'],
                name='Test Set',
                mode='lines',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                name='Forecast',
                mode='lines',
                line=dict(color='red', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title='Sales Forecast vs Actual',
                xaxis_title='Date',
                yaxis_title='Sales',
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("📈 Model Performance Metrics")
            
            test_forecast = forecast[forecast['ds'] <= df_test['ds'].max()].copy()
            if len(test_forecast) > 0:
                test_forecast = test_forecast.head(len(df_test))
                
                mae = mean_absolute_error(df_test['y'].values, test_forecast['yhat'].values)
                rmse = np.sqrt(mean_squared_error(df_test['y'].values, test_forecast['yhat'].values))
                mape = mean_absolute_percentage_error(df_test['y'].values, test_forecast['yhat'].values)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("MAPE", f"{mape:.2%}")
            
            # Download forecast
            st.subheader("💾 Download Results")
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Upload a CSV file to get started. Your data should have at least 'Date' and 'Sales' columns.")
    st.markdown("""
    ### 📝 Sample Data Format:
    
    | Date | Sales | Promo |
    |------|-------|-------|
    | 2023-01-01 | 1500 | 0 |
    | 2023-01-02 | 1600 | 0 |
    | 2023-01-03 | 2000 | 1 |
    
    ### ✨ Features:
    - **Prophet**: Great for data with strong seasonal patterns
    - **ARIMA**: Ideal for stationary time series
    - **LSTM**: Powerful for complex non-linear patterns
    """)
