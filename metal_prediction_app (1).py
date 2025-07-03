# Metal Price Prediction App - Streamlit Cloud Compatible
# This version only uses packages available on Streamlit Cloud by default

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Metal Price Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MetalPricePredictor:
    def __init__(self):
        self.metals = ['Gold', 'Silver', 'Copper', 'Aluminum', 'Steel']
        self.data = {}
        
    def generate_sample_data(self, metal, start_date='2000-01-01', end_date='2025-01-01'):
        """Generate realistic sample data for metal prices"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base prices and volatility for each metal (in INR)
        base_prices = {
            'Gold': 30000,      # per 10g
            'Silver': 60000,    # per kg
            'Copper': 700000,   # per tonne
            'Aluminum': 180000, # per tonne
            'Steel': 50000      # per tonne
        }
        
        volatilities = {
            'Gold': 0.02,
            'Silver': 0.03,
            'Copper': 0.025,
            'Aluminum': 0.02,
            'Steel': 0.015
        }
        
        # Set seed for reproducible results
        np.random.seed(42)
        base_price = base_prices[metal]
        volatility = volatilities[metal]
        
        # Create trend (gradual increase over time)
        trend = np.linspace(0, 0.8, len(date_range))
        
        # Create seasonality (annual cycle)
        seasonality = 0.15 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
        
        # Create random walk for realistic price movement
        random_walk = np.cumsum(np.random.normal(0, volatility, len(date_range)))
        
        # Add some market shocks (sudden price changes)
        shocks = np.zeros(len(date_range))
        shock_indices = np.random.choice(len(date_range), size=50, replace=False)
        shocks[shock_indices] = np.random.normal(0, volatility * 5, 50)
        
        # Combine all components
        prices = base_price * (1 + trend + seasonality + random_walk + np.cumsum(shocks * 0.1))
        
        # Ensure prices are positive
        prices = np.maximum(prices, base_price * 0.5)
        
        # Create comprehensive dataset
        df = pd.DataFrame({
            'Date': date_range,
            'Price': prices,
            'Volume': np.random.normal(1000, 200, len(date_range)),
            'USD_INR': np.random.normal(75, 5, len(date_range)),
            'Oil_Price': np.random.normal(80, 20, len(date_range)),
            'Gold_Silver_Ratio': np.random.normal(70, 10, len(date_range)) if metal != 'Gold' else None
        })
        
        # Add technical indicators
        df['MA_7'] = df['Price'].rolling(window=7).mean()
        df['MA_30'] = df['Price'].rolling(window=30).mean()
        df['MA_90'] = df['Price'].rolling(window=90).mean()
        df['Price_Change'] = df['Price'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=30).std()
        df['RSI'] = self.calculate_rsi(df['Price'])
        
        # Remove NaN values
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def simple_linear_regression(self, X, y):
        """Simple linear regression implementation"""
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        return slope, intercept
    
    def predict_with_multiple_methods(self, df, days=30):
        """Predict using multiple simple methods and ensemble them"""
        predictions_dict = {}
        
        # Method 1: Moving Average Trend
        ma_trend = (df['MA_7'].iloc[-1] - df['MA_7'].iloc[-30]) / 30
        ma_predictions = [df['Price'].iloc[-1] + ma_trend * i for i in range(1, days + 1)]
        predictions_dict['MA_Trend'] = ma_predictions
        
        # Method 2: Linear Regression on recent data
        recent_data = df.tail(60)
        X = np.arange(len(recent_data))
        y = recent_data['Price'].values
        slope, intercept = self.simple_linear_regression(X, y)
        
        start_x = len(recent_data)
        lr_predictions = [slope * (start_x + i) + intercept for i in range(1, days + 1)]
        predictions_dict['Linear_Regression'] = lr_predictions
        
        # Method 3: Seasonal Decomposition
        # Extract seasonal pattern from historical data
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        seasonal_avg = df.groupby('Day_of_Year')['Price'].mean()
        
        seasonal_predictions = []
        for i in range(1, days + 1):
            future_date = df['Date'].iloc[-1] + timedelta(days=i)
            day_of_year = future_date.timetuple().tm_yday
            seasonal_factor = seasonal_avg.get(day_of_year, seasonal_avg.mean())
            recent_avg = df['Price'].tail(30).mean()
            seasonal_pred = recent_avg * (seasonal_factor / seasonal_avg.mean())
            seasonal_predictions.append(seasonal_pred)
        
        predictions_dict['Seasonal'] = seasonal_predictions
        
        # Method 4: Volatility-adjusted prediction
        volatility = df['Volatility'].iloc[-1]
        base_price = df['Price'].iloc[-1]
        vol_predictions = []
        
        for i in range(1, days + 1):
            # Add some randomness based on volatility
            np.random.seed(42 + i)  # For reproducibility
            vol_adj = np.random.normal(0, volatility * base_price)
            vol_pred = base_price + (ma_trend * i) + vol_adj
            vol_predictions.append(vol_pred)
        
        predictions_dict['Volatility_Adjusted'] = vol_predictions
        
        # Ensemble: Average all methods
        ensemble_predictions = []
        for i in range(days):
            avg_pred = np.mean([
                predictions_dict['MA_Trend'][i],
                predictions_dict['Linear_Regression'][i],
                predictions_dict['Seasonal'][i],
                predictions_dict['Volatility_Adjusted'][i]
            ])
            ensemble_predictions.append(avg_pred)
        
        # Create dates for predictions
        prediction_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
        
        # Calculate confidence intervals
        pred_std = np.std([predictions_dict[method] for method in predictions_dict], axis=0)
        lower_bound = np.array(ensemble_predictions) - 1.96 * pred_std
        upper_bound = np.array(ensemble_predictions) + 1.96 * pred_std
        
        return pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': ensemble_predictions,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'MA_Trend': predictions_dict['MA_Trend'],
            'Linear_Regression': predictions_dict['Linear_Regression'],
            'Seasonal': predictions_dict['Seasonal'],
            'Volatility_Adjusted': predictions_dict['Volatility_Adjusted']
        })
    
    def calculate_metrics(self, df):
        """Calculate various performance metrics"""
        # Simple backtest on last 30 days
        test_data = df.tail(30)
        train_data = df.iloc[:-30]
        
        # Simple prediction for backtesting
        trend = (train_data['Price'].iloc[-1] - train_data['Price'].iloc[-30]) / 30
        predicted_prices = [train_data['Price'].iloc[-1] + trend * i for i in range(1, 31)]
        actual_prices = test_data['Price'].values
        
        # Calculate metrics
        mae = np.mean(np.abs(predicted_prices - actual_prices))
        rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Accuracy': max(0, 100 - mape)
        }

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return MetalPricePredictor()

def main():
    st.markdown('<h1 class="main-header">🥇 Metal Price Prediction AI</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Predictions for Gold, Silver, Copper, Aluminum & Steel prices in Indian markets**")
    
    # Initialize predictor
    predictor = load_predictor()
    
    # Sidebar
    st.sidebar.header("🔧 Prediction Controls")
    
    # Metal selection
    selected_metal = st.sidebar.selectbox(
        "Select Metal:",
        predictor.metals,
        help="Choose which metal you want to analyze and predict"
    )
    
    # Prediction parameters
    pred_days = st.sidebar.slider(
        "Prediction Days:",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of days to predict into the future"
    )
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Quick Prediction", "Detailed Analysis", "Compare Methods"],
        help="Choose the type of analysis to perform"
    )
    
    # Generate data and analysis
    with st.spinner(f"Loading {selected_metal} data..."):
        df = predictor.generate_sample_data(selected_metal)
        metrics = predictor.calculate_metrics(df)
    
    # Main dashboard
    st.subheader(f"📊 {selected_metal} Price Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['Price'].iloc[-1]
        st.metric(
            "Current Price", 
            f"₹{current_price:,.2f}",
            help="Latest price in the dataset"
        )
    
    with col2:
        monthly_change = ((df['Price'].iloc[-1] - df['Price'].iloc[-30]) / df['Price'].iloc[-30]) * 100
        st.metric(
            "30-Day Change", 
            f"{monthly_change:+.2f}%",
            delta=f"{monthly_change:.2f}%"
        )
    
    with col3:
        st.metric(
            "Volatility", 
            f"{df['Volatility'].iloc[-1]:.4f}",
            help="30-day price volatility"
        )
    
    with col4:
        st.metric(
            "Model Accuracy", 
            f"{metrics['Accuracy']:.1f}%",
            help="Backtest accuracy on recent data"
        )
    
    # Historical price chart
    st.subheader("📈 Historical Price Analysis")
    
    # Create interactive chart
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_30'],
        mode='lines',
        name='30-Day MA',
        line=dict(color='#ff7f0e', width=1),
        hovertemplate='Date: %{x}<br>30-Day MA: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_90'],
        mode='lines',
        name='90-Day MA',
        line=dict(color='#2ca02c', width=1),
        hovertemplate='Date: %{x}<br>90-Day MA: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{selected_metal} Price History (2000-2025)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction section
    st.subheader("🔮 Price Predictions")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Generating AI predictions..."):
            predictions = predictor.predict_with_multiple_methods(df, pred_days)
        
        # Display prediction summary
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            next_day_price = predictions['Predicted_Price'].iloc[0]
            st.metric(
                "Next Day Prediction", 
                f"₹{next_day_price:,.2f}",
                delta=f"{((next_day_price - current_price) / current_price) * 100:+.2f}%"
            )
        
        with col2:
            end_period_price = predictions['Predicted_Price'].iloc[-1]
            st.metric(
                f"{pred_days}-Day Prediction", 
                f"₹{end_period_price:,.2f}",
                delta=f"{((end_period_price - current_price) / current_price) * 100:+.2f}%"
            )
        
        with col3:
            total_change = end_period_price - current_price
            st.metric(
                "Expected Change", 
                f"₹{total_change:+,.2f}",
                delta=f"{(total_change / current_price) * 100:+.2f}%"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction chart
        fig_pred = go.Figure()
        
        # Historical data (last 90 days)
        recent_data = df.tail(90)
        fig_pred.add_trace(go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Price'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Price'],
            mode='lines+markers',
            name='AI Prediction',
            line=dict(color='#ff4444', width=3),
            marker=dict(size=6)
        ))
        
        # Confidence intervals
        fig_pred.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Upper_Bound'],
            mode='lines',
            name='Upper Bound (95%)',
            line=dict(color='rgba(255,68,68,0.3)', width=1),
            showlegend=False
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Lower_Bound'],
            mode='lines',
            name='Lower Bound (95%)',
            line=dict(color='rgba(255,68,68,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,68,68,0.1)',
            showlegend=True
        ))
        
        fig_pred.update_layout(
            title=f"{selected_metal} Price Forecast - Next {pred_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Detailed analysis based on selection
        if analysis_type == "Detailed Analysis":
            st.subheader("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Technical Indicators**")
                st.write(f"RSI (14): {df['RSI'].iloc[-1]:.2f}")
                st.write(f"Current vs MA_30: {((current_price - df['MA_30'].iloc[-1]) / df['MA_30'].iloc[-1]) * 100:+.2f}%")
                st.write(f"Current vs MA_90: {((current_price - df['MA_90'].iloc[-1]) / df['MA_90'].iloc[-1]) * 100:+.2f}%")
                
                # Trend analysis
                if df['MA_7'].iloc[-1] > df['MA_30'].iloc[-1]:
                    st.success("📈 Short-term trend: BULLISH")
                else:
                    st.error("📉 Short-term trend: BEARISH")
            
            with col2:
                st.markdown("**🎯 Model Performance**")
                st.write(f"Mean Absolute Error: ₹{metrics['MAE']:,.2f}")
                st.write(f"Root Mean Square Error: ₹{metrics['RMSE']:,.2f}")
                st.write(f"Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")
                st.write(f"Prediction Accuracy: {metrics['Accuracy']:.1f}%")
        
        elif analysis_type == "Compare Methods":
            st.subheader("🔍 Prediction Methods Comparison")
            
            # Create comparison chart
            fig_compare = go.Figure()
            
            methods = ['MA_Trend', 'Linear_Regression', 'Seasonal', 'Volatility_Adjusted', 'Predicted_Price']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, method in enumerate(methods):
                name = method.replace('_', ' ').title()
                if method == 'Predicted_Price':
                    name = 'Ensemble (Final)'
                
                fig_compare.add_trace(go.Scatter(
                    x=predictions['Date'],
                    y=predictions[method],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i], width=2 if method == 'Predicted_Price' else 1),
                    marker=dict(size=6 if method == 'Predicted_Price' else 4)
                ))
            
            fig_compare.update_layout(
                title="Comparison of Different Prediction Methods",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Prediction table
        with st.expander("📋 Detailed Prediction Table"):
            display_predictions = predictions[['Date', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']].copy()
            display_predictions['Predicted_Price'] = display_predictions['Predicted_Price'].round(2)
            display_predictions['Lower_Bound'] = display_predictions['Lower_Bound'].round(2)
            display_predictions['Upper_Bound'] = display_predictions['Upper_Bound'].round(2)
            
            st.dataframe(display_predictions, use_container_width=True)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **Metal Price Prediction AI** uses advanced algorithms to forecast metal prices based on:
    
    - Historical price patterns
    - Technical indicators
    - Seasonal trends
    - Market volatility
    - Multiple prediction methods
    
    **Disclaimer:** This is for educational purposes. Real trading decisions should involve professional financial advice.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Built with Streamlit • Made with ❤️ in India**")
    st.markdown("*This application uses simulated data for demonstration. For production use, integrate with real metal price APIs.*")

if __name__ == "__main__":
    main()