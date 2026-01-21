"""
Indian Stock Analyzer - Streamlit Dashboard
BSE & NSE Stock Market Analysis and Price Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Analyzer Pro - BSE & NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #FF9933 0%, #138808 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-moderate { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .inr-price { font-size: 1.5rem; font-weight: bold; color: #138808; }
    .stock-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF9933;
    }
    .bullish { color: #28a745; }
    .bearish { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Import local modules
import sys
sys.path.insert(0, '.')

from config import DISCLAIMER, DEFAULT_PREDICTION_DAYS, DEFAULT_YEARS_HISTORY
from data.fetcher import StockDataFetcher
from data.preprocessor import DataPreprocessor
from features.indicators import TechnicalIndicators
from analysis.eda import ExploratoryDataAnalysis
from analysis.risk import RiskAnalyzer
from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from forecasting.predictor import ForecastEngine, create_combined_forecast_chart

# Indian Stock Data
NIFTY_50_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TATASTEEL.NS": "Tata Steel",
    "SUNPHARMA.NS": "Sun Pharma",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ONGC.NS": "ONGC",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid",
    "TATAMOTORS.NS": "Tata Motors",
    "HCLTECH.NS": "HCL Technologies",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "TECHM.NS": "Tech Mahindra",
    "NESTLEIND.NS": "Nestle India"
}

BSE_STOCKS = {
    "RELIANCE.BO": "Reliance Industries",
    "TCS.BO": "Tata Consultancy Services",
    "HDFCBANK.BO": "HDFC Bank",
    "INFY.BO": "Infosys",
    "ICICIBANK.BO": "ICICI Bank",
    "SBIN.BO": "State Bank of India",
    "BHARTIARTL.BO": "Bharti Airtel",
    "ITC.BO": "ITC Limited",
    "LT.BO": "Larsen & Toubro",
    "WIPRO.BO": "Wipro"
}

SECTOR_INDICES = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "BSE SENSEX",
    "^NSEBANK": "NIFTY Bank",
    "^CNXIT": "NIFTY IT",
    "^CNXAUTO": "NIFTY Auto",
    "^CNXPHARMA": "NIFTY Pharma"
}


def create_volatility_cone_chart(df, symbol):
    """Create volatility cone chart showing volatility across different time periods."""
    fig = go.Figure()
    
    returns = df['daily_return'].dropna()
    
    periods = [5, 10, 20, 30, 60, 90, 120, 252]
    period_labels = ['1W', '2W', '1M', '1.5M', '3M', '4.5M', '6M', '1Y']
    
    current_vols = []
    min_vols = []
    max_vols = []
    mean_vols = []
    
    for period in periods:
        if len(returns) >= period:
            rolling_vol = returns.rolling(period).std() * np.sqrt(252) * 100
            current_vols.append(rolling_vol.iloc[-1])
            min_vols.append(rolling_vol.min())
            max_vols.append(rolling_vol.max())
            mean_vols.append(rolling_vol.mean())
        else:
            current_vols.append(None)
            min_vols.append(None)
            max_vols.append(None)
            mean_vols.append(None)
    
    # Max volatility (top of cone)
    fig.add_trace(go.Scatter(
        x=period_labels, y=max_vols,
        mode='lines+markers',
        name='Max Volatility',
        line=dict(color='#ef5350', width=2, dash='dash')
    ))
    
    # Mean volatility
    fig.add_trace(go.Scatter(
        x=period_labels, y=mean_vols,
        mode='lines+markers',
        name='Mean Volatility',
        line=dict(color='#ff9800', width=2)
    ))
    
    # Min volatility (bottom of cone)
    fig.add_trace(go.Scatter(
        x=period_labels, y=min_vols,
        mode='lines+markers',
        name='Min Volatility',
        line=dict(color='#26a69a', width=2, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255, 152, 0, 0.1)'
    ))
    
    # Current volatility
    fig.add_trace(go.Scatter(
        x=period_labels, y=current_vols,
        mode='lines+markers',
        name='Current Volatility',
        line=dict(color='#2196f3', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f'{symbol} Volatility Cone Analysis',
        yaxis_title='Annualized Volatility (%)',
        xaxis_title='Time Period',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_risk_gauge(risk_score, risk_level):
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#28a745'},
                {'range': [25, 50], 'color': '#ffc107'},
                {'range': [50, 75], 'color': '#fd7e14'},
                {'range': [75, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


def create_var_chart(df, symbol):
    """Create Value at Risk visualization."""
    returns = df['daily_return'].dropna()
    
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Daily Returns',
        marker_color='rgba(33, 150, 243, 0.7)',
        opacity=0.7
    ))
    
    # VaR lines
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100
    
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                  annotation_text=f"VaR 95%: {var_95:.2f}%")
    fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                  annotation_text=f"VaR 99%: {var_99:.2f}%")
    fig.add_vline(x=0, line_dash="dot", line_color="white")
    
    fig.update_layout(
        title=f'{symbol} Value at Risk (VaR) Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=350
    )
    
    return fig


def create_rolling_risk_chart(df, symbol):
    """Create rolling risk metrics chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Rolling Volatility (20-day)', 'Rolling Sharpe Ratio', 'Rolling Max Drawdown'),
        row_heights=[0.35, 0.35, 0.3]
    )
    
    returns = df['daily_return'].dropna()
    
    # Rolling volatility
    rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_vol,
        mode='lines', name='20-day Volatility',
        line=dict(color='#ff9800', width=2)
    ), row=1, col=1)
    
    # Rolling Sharpe
    rolling_sharpe = (returns.rolling(60).mean() * 252) / (returns.rolling(60).std() * np.sqrt(252))
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_sharpe,
        mode='lines', name='60-day Sharpe',
        line=dict(color='#2196f3', width=2)
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Rolling max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(60, min_periods=1).max()
    drawdown = ((cumulative - rolling_max) / rolling_max) * 100
    fig.add_trace(go.Scatter(
        x=df['date'], y=drawdown,
        mode='lines', name='Rolling Drawdown',
        line=dict(color='#ef5350', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 83, 80, 0.2)'
    ), row=3, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_beta_correlation_chart(df, benchmark_df, symbol):
    """Create beta and correlation analysis chart."""
    if benchmark_df is None or benchmark_df.empty:
        return None
    
    # Merge data
    merged = pd.merge(
        df[['date', 'daily_return']].rename(columns={'daily_return': 'stock_return'}),
        benchmark_df[['date', 'daily_return']].rename(columns={'daily_return': 'benchmark_return'}),
        on='date',
        how='inner'
    ).dropna()
    
    if len(merged) < 30:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Stock vs NIFTY 50 Returns', 'Rolling Beta (60-day)')
    )
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=merged['benchmark_return'] * 100,
        y=merged['stock_return'] * 100,
        mode='markers',
        name='Daily Returns',
        marker=dict(color='rgba(33, 150, 243, 0.5)', size=5)
    ), row=1, col=1)
    
    # Regression line
    slope, intercept = np.polyfit(merged['benchmark_return'], merged['stock_return'], 1)
    x_line = np.array([merged['benchmark_return'].min(), merged['benchmark_return'].max()])
    fig.add_trace(go.Scatter(
        x=x_line * 100,
        y=(slope * x_line + intercept) * 100,
        mode='lines',
        name=f'Beta = {slope:.2f}',
        line=dict(color='#ff9800', width=2)
    ), row=1, col=1)
    
    # Rolling beta
    cov = merged['stock_return'].rolling(60).cov(merged['benchmark_return'])
    var = merged['benchmark_return'].rolling(60).var()
    rolling_beta = cov / var
    
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=rolling_beta,
        mode='lines',
        name='Rolling Beta',
        line=dict(color='#2196f3', width=2)
    ), row=1, col=2)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=2)
    
    fig.update_layout(
        height=350,
        template='plotly_dark',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="NIFTY 50 Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Stock Return (%)", row=1, col=1)
    
    return fig


def create_sector_comparison(df, symbol, sector_name="NIFTY 50"):
    """Create sector performance comparison."""
    fig = go.Figure()
    
    # Normalize prices
    normalized = (df['close'] / df['close'].iloc[0] - 1) * 100
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=normalized,
        mode='lines',
        name=symbol,
        line=dict(color='#2196f3', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=350
    )
    
    return fig


def create_monthly_returns_heatmap(df, symbol):
    """Create monthly returns heatmap."""
    df_copy = df.copy()
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['month'] = df_copy['date'].dt.month
    
    # Calculate monthly returns
    monthly = df_copy.groupby(['year', 'month'])['close'].agg(['first', 'last'])
    monthly['return'] = ((monthly['last'] / monthly['first']) - 1) * 100
    monthly = monthly.reset_index()
    
    # Pivot for heatmap
    pivot = monthly.pivot(index='year', columns='month', values='return')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names[:pivot.shape[1]],
        y=pivot.index.astype(str),
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title='Return %')
    ))
    
    fig.update_layout(
        title=f'{symbol} Monthly Returns Heatmap',
        xaxis_title='Month',
        yaxis_title='Year',
        template='plotly_dark',
        height=400
    )
    
    return fig


def format_inr(value):
    """Format value in Indian Rupee format."""
    if isinstance(value, str):
        return value
    if value >= 1e7:  # Crore
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:  # Lakh
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.2f}"


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">📈 Stock Analyzer Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Market Analysis & AI Predictions by Aryan</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%); 
                    padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <p style="color: #333; font-weight: bold; margin: 0; text-align: center;">
                ✨ Analyze ANY BSE or NSE Stock!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stock symbol input - PRIMARY METHOD
        st.subheader("🔍 Enter Stock Symbol")
        
        symbol_input = st.text_input(
            "Stock Symbol",
            value="RELIANCE",
            help="Enter any BSE or NSE stock symbol (e.g., RELIANCE, TCS, INFY, ZOMATO, PAYTM)"
        ).upper().strip()
        
        # Exchange selection
        exchange = st.radio(
            "Select Exchange",
            ["NSE (National Stock Exchange)", "BSE (Bombay Stock Exchange)"],
            horizontal=False,
            help="NSE uses .NS suffix, BSE uses .BO suffix"
        )
        
        suffix = ".NS" if "NSE" in exchange else ".BO"
        
        # Auto-detect if user already added suffix
        if symbol_input.endswith('.NS') or symbol_input.endswith('.BO'):
            symbol = symbol_input
        else:
            symbol = f"{symbol_input}{suffix}"
        
        st.success(f"📌 Analyzing: **{symbol}**")
        
        # Quick pick popular stocks
        with st.expander("📈 Quick Pick - Popular Stocks"):
            col1, col2 = st.columns(2)
            
            popular_nse = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
                          "SBIN", "BHARTIARTL", "ITC", "WIPRO", "TATAMOTORS",
                          "ZOMATO", "PAYTM", "NYKAA", "IRCTC", "ADANIENT",
                          "TATASTEEL", "BAJFINANCE", "MARUTI", "LT", "HINDUNILVR"]
            
            with col1:
                st.markdown("**NSE Stocks:**")
                for stock in popular_nse[:10]:
                    if st.button(stock, key=f"nse_{stock}", use_container_width=True):
                        st.session_state['quick_pick'] = f"{stock}.NS"
                        st.rerun()
            
            with col2:
                st.markdown("**More NSE:**")
                for stock in popular_nse[10:20]:
                    if st.button(stock, key=f"nse2_{stock}", use_container_width=True):
                        st.session_state['quick_pick'] = f"{stock}.NS"
                        st.rerun()
        
        # Check for quick pick
        if 'quick_pick' in st.session_state:
            symbol = st.session_state['quick_pick']
            del st.session_state['quick_pick']
        
        # Symbol format help
        with st.expander("ℹ️ Symbol Guide"):
            st.markdown("""
            **How to find your stock symbol:**
            
            | Exchange | Suffix | Example |
            |----------|--------|---------|
            | NSE | `.NS` | RELIANCE.NS |
            | BSE | `.BO` | RELIANCE.BO |
            
            **Popular Symbols:**
            - **Large Cap:** RELIANCE, TCS, HDFCBANK, INFY
            - **Banks:** SBIN, ICICIBANK, KOTAKBANK, AXISBANK
            - **IT:** TCS, INFY, WIPRO, HCLTECH, TECHM
            - **Auto:** TATAMOTORS, MARUTI, BAJAJ-AUTO
            - **New Age:** ZOMATO, PAYTM, NYKAA, POLICYBZR
            - **PSU:** IRCTC, COALINDIA, ONGC, NTPC
            
            **Tip:** Just enter the company name without suffix, 
            we'll add it automatically based on your exchange selection!
            """)
        
        st.divider()
        
        # Date range
        st.subheader("📅 Date Range")
        years = st.slider(
            "Years of Historical Data",
            min_value=1,
            max_value=10,
            value=3
        )

        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        st.caption(f"From: {start_date.strftime('%Y-%m-%d')}")
        st.caption(f"To: {end_date.strftime('%Y-%m-%d')}")
        
        # Prediction settings
        st.subheader("🔮 Prediction Settings")
        prediction_days = st.slider(
            "Forecast Horizon (Days)",
            min_value=7,
            max_value=90,
            value=30
        )
        
        model_choice = st.selectbox(
            "Forecasting Model",
            ["Prophet", "XGBoost", "Compare All"],
            help="Select the model for price prediction"
        )
        
        # Compare with index
        compare_index = st.checkbox("Compare with NIFTY 50", value=True)
        
        # Analyze button
        analyze_btn = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
        
        # Disclaimer
        st.divider()
        with st.expander("⚠️ Disclaimer", expanded=False):
            st.markdown(DISCLAIMER)
            
        st.markdown("---")
        st.markdown("Developed with ❤️ by **Aryan**")
    
    # Main content
    if analyze_btn and symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            df, error = StockDataFetcher.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Fetch NIFTY 50 for comparison
            benchmark_df = None
            if compare_index:
                benchmark_df, _ = StockDataFetcher.fetch_stock_data(
                    symbol="^NSEI",
                    start_date=start_date,
                    end_date=end_date
                )
                if benchmark_df is not None and not benchmark_df.empty:
                    benchmark_df, _ = DataPreprocessor.preprocess(benchmark_df)
                    benchmark_df = TechnicalIndicators.add_all_indicators(benchmark_df)
        
        if error:
            st.error(f"❌ {error}")
            st.info(f"""
            **Tips for Indian Stocks:**
            - NSE stocks: Add `.NS` suffix (e.g., `RELIANCE.NS`)
            - BSE stocks: Add `.BO` suffix (e.g., `RELIANCE.BO`)
            - Check if the symbol is correct on Yahoo Finance
            """)
            return
        
        if df is None or df.empty:
            st.error("❌ No data available for this symbol.")
            return
        
        # Preprocess data
        with st.spinner("Processing data..."):
            df, preprocess_report = DataPreprocessor.preprocess(df)
            df = TechnicalIndicators.add_all_indicators(df)
        
        # Get stock info
        stock_info = StockDataFetcher.get_stock_info(symbol)
        
        # Display stock info header
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        
        with col1:
            stock_name = NIFTY_50_STOCKS.get(symbol, BSE_STOCKS.get(symbol, symbol.replace('.NS', '').replace('.BO', '')))
            st.metric(
                label=stock_name,
                value=f"₹{current_price:,.2f}",
                delta=f"{change_pct:+.2f}%"
            )
        
        with col2:
            st.metric("Day High", f"₹{df['high'].iloc[-1]:,.2f}")
        
        with col3:
            st.metric("Day Low", f"₹{df['low'].iloc[-1]:,.2f}")
        
        with col4:
            st.metric("52-Week High", f"₹{df['high'].max():,.2f}")
        
        with col5:
            st.metric("52-Week Low", f"₹{df['low'].min():,.2f}")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Price Analysis",
            "📉 Technical Indicators", 
            "⚠️ Risk Analysis",
            "📈 Advanced Risk Charts",
            "🔮 Price Forecast",
            "📋 Statistics"
        ])
        
        # Tab 1: Price Analysis
        with tab1:
            st.subheader("📊 Historical Price Chart")
            
            chart_type = st.radio(
                "Chart Type",
                ["Candlestick", "Line"],
                horizontal=True,
                key="price_chart_type"
            )
            
            if chart_type == "Candlestick":
                fig = ExploratoryDataAnalysis.create_price_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
            else:
                fig = ExploratoryDataAnalysis.create_line_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly returns heatmap
            st.subheader("📅 Monthly Returns Heatmap")
            fig_heatmap = create_monthly_returns_heatmap(df, symbol.replace('.NS', '').replace('.BO', ''))
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Returns distribution & Drawdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Returns Distribution")
                fig_returns = ExploratoryDataAnalysis.create_returns_distribution(df, symbol.replace('.NS', '').replace('.BO', ''))
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with col2:
                st.subheader("📉 Drawdown Analysis")
                fig_dd = ExploratoryDataAnalysis.create_drawdown_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
                st.plotly_chart(fig_dd, use_container_width=True)
        
        # Tab 2: Technical Indicators
        with tab2:
            st.subheader("📉 Technical Indicators")
            
            # Bollinger Bands
            if 'bb_upper' in df.columns:
                st.markdown("### Bollinger Bands")
                fig_bb = ExploratoryDataAnalysis.create_bollinger_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
                st.plotly_chart(fig_bb, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'rsi' in df.columns:
                    st.markdown("### RSI (Relative Strength Index)")
                    fig_rsi = ExploratoryDataAnalysis.create_rsi_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    current_rsi = df['rsi'].iloc[-1]
                    if current_rsi > 70:
                        st.error(f"⚠️ RSI = {current_rsi:.1f} (Overbought - Potential Sell Signal)")
                    elif current_rsi < 30:
                        st.success(f"✅ RSI = {current_rsi:.1f} (Oversold - Potential Buy Signal)")
                    else:
                        st.info(f"ℹ️ RSI = {current_rsi:.1f} (Neutral Zone)")
            
            with col2:
                if 'macd' in df.columns:
                    st.markdown("### MACD")
                    fig_macd = ExploratoryDataAnalysis.create_macd_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    macd_val = df['macd'].iloc[-1]
                    signal_val = df['macd_signal'].iloc[-1]
                    if macd_val > signal_val:
                        st.success("✅ MACD above Signal Line (Bullish Crossover)")
                    else:
                        st.warning("⚠️ MACD below Signal Line (Bearish Crossover)")
            
            # Moving Averages Summary
            st.markdown("### Moving Averages Summary")
            ma_cols = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200']
            ma_data = []
            for col in ma_cols:
                if col in df.columns:
                    ma_val = df[col].iloc[-1]
                    signal = "🟢 Above" if current_price > ma_val else "🔴 Below"
                    ma_data.append({
                        'Moving Average': col.upper().replace('_', ' '),
                        'Value': f"₹{ma_val:,.2f}",
                        'Current Price Position': signal,
                        'Difference': f"{((current_price - ma_val) / ma_val * 100):+.2f}%"
                    })
            
            if ma_data:
                st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)
        
        # Tab 3: Risk Analysis
        with tab3:
            st.subheader("⚠️ Comprehensive Risk Assessment")
            
            with st.spinner("Calculating risk metrics..."):
                risk_summary = RiskAnalyzer.get_risk_summary(df)
            
            # Risk Score Gauge
            col1, col2, col3 = st.columns([1, 1.5, 1])
            
            with col2:
                fig_gauge = create_risk_gauge(risk_summary['risk_score'], risk_summary['risk_level'])
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                risk_level = risk_summary['risk_level']
                if risk_level in ['High', 'Very High']:
                    st.error(f"🔴 **{risk_level} Risk** - Exercise caution!")
                elif risk_level == 'Moderate':
                    st.warning(f"🟡 **{risk_level} Risk** - Monitor closely")
                else:
                    st.success(f"🟢 **{risk_level} Risk** - Relatively stable")
            
            # Risk Warnings
            if risk_summary['warnings']:
                st.markdown("### ⚠️ Active Risk Alerts")
                for warning in risk_summary['warnings']:
                    if warning['level'] == 'high':
                        st.error(warning['message'])
                    elif warning['level'] == 'medium':
                        st.warning(warning['message'])
                    else:
                        st.info(warning['message'])
            
            # Risk Metrics Cards
            st.markdown("### 📊 Risk Metrics Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            vol_metrics = risk_summary['volatility_metrics']
            var_metrics = risk_summary['var_metrics']
            trend = risk_summary['trend_metrics']
            
            with col1:
                st.markdown("""
                <div class="stock-card">
                    <h4>📈 Volatility</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Daily", f"{vol_metrics['daily_volatility']*100:.2f}%")
                st.metric("Annualized", f"{vol_metrics['annualized_volatility']*100:.1f}%")
                st.metric("Percentile", f"{vol_metrics['volatility_percentile']:.0f}th")
            
            with col2:
                st.markdown("""
                <div class="stock-card">
                    <h4>💰 Value at Risk</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("1-Day VaR (95%)", f"{var_metrics['var_1day_pct']:.2f}%")
                st.metric("CVaR", f"{var_metrics['cvar_pct']:.2f}%")
                st.metric("Per ₹1L Investment", f"₹{var_metrics['var_1day_amount']*10:,.0f}")
            
            with col3:
                st.markdown("""
                <div class="stock-card">
                    <h4>📉 Trend Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Direction", trend['trend_direction'])
                st.metric("Strength", trend['trend_strength'])
                st.metric("R-Squared", f"{trend['r_squared']:.3f}")
            
            with col4:
                st.markdown("""
                <div class="stock-card">
                    <h4>📊 Drawdown</h4>
                </div>
                """, unsafe_allow_html=True)
                stats = ExploratoryDataAnalysis.get_summary_statistics(df)
                st.metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%")
                st.metric("Downside Vol", f"{vol_metrics['downside_volatility']*100:.1f}%")
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
        
        # Tab 4: Advanced Risk Charts
        with tab4:
            st.subheader("📈 Advanced Risk Visualizations")
            
            # Volatility Cone
            st.markdown("### 🔔 Volatility Cone Analysis")
            st.caption("Shows how current volatility compares to historical ranges across different time periods")
            fig_vol_cone = create_volatility_cone_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
            st.plotly_chart(fig_vol_cone, use_container_width=True)
            
            # VaR Distribution
            st.markdown("### 💰 Value at Risk (VaR) Distribution")
            st.caption("Histogram of daily returns with VaR thresholds marked")
            fig_var = create_var_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Rolling Risk Metrics
            st.markdown("### 📉 Rolling Risk Metrics Over Time")
            fig_rolling = create_rolling_risk_chart(df, symbol.replace('.NS', '').replace('.BO', ''))
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            # Beta Analysis (if benchmark available)
            if benchmark_df is not None:
                st.markdown("### 📊 Beta & Correlation with NIFTY 50")
                fig_beta = create_beta_correlation_chart(df, benchmark_df, symbol.replace('.NS', '').replace('.BO', ''))
                if fig_beta:
                    st.plotly_chart(fig_beta, use_container_width=True)
                    
                    # Calculate beta
                    merged = pd.merge(
                        df[['date', 'daily_return']].rename(columns={'daily_return': 'stock'}),
                        benchmark_df[['date', 'daily_return']].rename(columns={'daily_return': 'benchmark'}),
                        on='date'
                    ).dropna()
                    
                    if len(merged) > 30:
                        cov = merged['stock'].cov(merged['benchmark'])
                        var = merged['benchmark'].var()
                        beta = cov / var
                        corr = merged['stock'].corr(merged['benchmark'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Beta", f"{beta:.2f}", help="Stock's sensitivity to market movements")
                        with col2:
                            st.metric("Correlation", f"{corr:.2f}", help="Correlation with NIFTY 50")
                        with col3:
                            if beta > 1.2:
                                st.warning("⚠️ High Beta - More volatile than market")
                            elif beta < 0.8:
                                st.success("✅ Low Beta - Less volatile than market")
                            else:
                                st.info("ℹ️ Beta ~1 - Moves with market")
            
            # Cumulative Returns
            st.markdown("### 📈 Cumulative Returns Chart")
            fig_cumret = create_sector_comparison(df, symbol.replace('.NS', '').replace('.BO', ''))
            st.plotly_chart(fig_cumret, use_container_width=True)
        
        # Tab 5: Price Forecast
        with tab5:
            st.subheader(f"🔮 {prediction_days}-Day Price Forecast")
            
            with st.spinner(f"Training {model_choice} model..."):
                try:
                    if model_choice == "Compare All":
                        models = [ProphetModel(), XGBoostModel()]
                        forecasts = {}
                        
                        for model in models:
                            try:
                                model.fit(df)
                                forecast = model.predict(periods=prediction_days)
                                forecasts[model.name] = forecast
                            except Exception as e:
                                st.warning(f"{model.name} error: {str(e)}")
                        
                        if forecasts:
                            fig = create_combined_forecast_chart(df, forecasts, symbol.replace('.NS', '').replace('.BO', ''))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("📊 Model Predictions Comparison")
                            summary_data = []
                            for model_name, forecast in forecasts.items():
                                last_pred = forecast['predicted'].iloc[-1]
                                change_pct = ((last_pred - current_price) / current_price) * 100
                                summary_data.append({
                                    'Model': model_name,
                                    'Current Price': f"₹{current_price:,.2f}",
                                    f'{prediction_days}-Day Target': f"₹{last_pred:,.2f}",
                                    'Expected Change': f"{change_pct:+.2f}%",
                                    'Signal': '🟢 Bullish' if change_pct > 2 else '🔴 Bearish' if change_pct < -2 else '🟡 Neutral'
                                })
                            
                            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    else:
                        if model_choice == "Prophet":
                            model = ProphetModel()
                        else:
                            model = XGBoostModel()
                        
                        engine = ForecastEngine(model)
                        forecast = engine.generate_forecast(df, periods=prediction_days)
                        
                        fig = engine.create_forecast_chart(symbol.replace('.NS', '').replace('.BO', ''))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        summary = engine.get_forecast_summary()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"₹{current_price:,.2f}")
                        with col2:
                            st.metric(f"{prediction_days}-Day Target", 
                                     f"₹{summary['final_prediction']:,.2f}",
                                     f"{summary['predicted_change_pct']:+.2f}%")
                        with col3:
                            direction = summary['predicted_direction']
                            st.metric("Trend", f"{'📈' if direction == 'Bullish' else '📉'} {direction}")
                        with col4:
                            if 'confidence_range' in summary:
                                st.metric("95% Range", 
                                         f"₹{summary['confidence_range']['lower']:,.0f} - ₹{summary['confidence_range']['upper']:,.0f}")
                        
                        with st.expander("📋 View Detailed Day-by-Day Predictions"):
                            pred_table = engine.get_prediction_table()
                            pred_table['predicted'] = pred_table['predicted'].apply(lambda x: f"₹{x:,.2f}")
                            if 'lower_bound' in pred_table.columns:
                                pred_table['lower_bound'] = pred_table['lower_bound'].apply(lambda x: f"₹{x:,.2f}")
                                pred_table['upper_bound'] = pred_table['upper_bound'].apply(lambda x: f"₹{x:,.2f}")
                            st.dataframe(pred_table, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Forecasting error: {str(e)}")
            
            st.warning("""
            ⚠️ **Important Disclaimer**: These predictions are probabilistic estimates based on historical patterns. 
            They are NOT investment advice. Stock markets are influenced by many external factors not captured by these models.
            Always consult a SEBI registered financial advisor before making investment decisions.
            """)
        
        # Tab 6: Statistics  
        with tab6:
            st.subheader("📋 Comprehensive Statistics")
            
            stats = ExploratoryDataAnalysis.get_summary_statistics(df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 💰 Price Statistics")
                price_stats_df = pd.DataFrame({
                    'Metric': ['Current Price', 'Start Price', '52-Week High', '52-Week Low', 'Total Return'],
                    'Value': [
                        f"₹{stats['current_price']:,.2f}",
                        f"₹{stats['start_price']:,.2f}",
                        f"₹{stats['max_price']:,.2f}",
                        f"₹{stats['min_price']:,.2f}",
                        f"{stats['total_return']:+.2f}%"
                    ]
                })
                st.dataframe(price_stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 📈 Return Statistics")
                return_stats_df = pd.DataFrame({
                    'Metric': ['Mean Daily Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    'Value': [
                        f"{stats['mean_daily_return']:.3f}%",
                        f"{stats['annualized_return']:+.2f}%",
                        f"{stats['annualized_volatility']:.2f}%",
                        f"{stats['sharpe_ratio']:.2f}",
                        f"{stats['max_drawdown']:.2f}%"
                    ]
                })
                st.dataframe(return_stats_df, use_container_width=True, hide_index=True)
            
            with col3:
                st.markdown("### 📊 Distribution Statistics")
                dist_stats_df = pd.DataFrame({
                    'Metric': ['Positive Days', 'Negative Days', 'Win Rate', 'Skewness', 'Kurtosis'],
                    'Value': [
                        f"{stats['positive_days']}",
                        f"{stats['negative_days']}",
                        f"{stats['positive_days']/(stats['positive_days']+stats['negative_days'])*100:.1f}%",
                        f"{stats['skewness']:.2f}",
                        f"{stats['kurtosis']:.2f}"
                    ]
                })
                st.dataframe(dist_stats_df, use_container_width=True, hide_index=True)
            
            # Data Period
            st.markdown("### 📅 Data Period")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Start Date", stats['start_date'])
            with col2:
                st.metric("End Date", stats['end_date'])
            with col3:
                st.metric("Trading Days", stats['trading_days'])
            
            # Volume Analysis
            if 'volume' in df.columns:
                st.markdown("### 📊 Volume Analysis")
                avg_volume = df['volume'].mean()
                recent_volume = df['volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Volume", f"{avg_volume/1e6:.2f}M")
                with col2:
                    st.metric("Recent 20-Day Avg", f"{recent_volume/1e6:.2f}M")
                with col3:
                    st.metric("Volume Trend", f"{volume_ratio:.2f}x average",
                             delta="Higher" if volume_ratio > 1.2 else "Lower" if volume_ratio < 0.8 else "Normal")
            
            # Raw Data
            with st.expander("📦 View Raw Data"):
                display_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(100).copy()
                display_df['open'] = display_df['open'].apply(lambda x: f"₹{x:,.2f}")
                display_df['high'] = display_df['high'].apply(lambda x: f"₹{x:,.2f}")
                display_df['low'] = display_df['low'].apply(lambda x: f"₹{x:,.2f}")
                display_df['close'] = display_df['close'].apply(lambda x: f"₹{x:,.2f}")
                display_df['volume'] = display_df['volume'].apply(lambda x: f"{x/1e6:.2f}M")
                st.dataframe(display_df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Data (CSV)",
                    data=csv,
                    file_name=f"{symbol}_stock_data.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: #333; margin: 0;">🇮🇳 Analyze ANY BSE or NSE Stock!</h2>
            <p style="color: #555; margin: 10px 0 0 0;">Enter any stock symbol in the sidebar and click Analyze</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            
            - **📊 Price Analysis**: Interactive candlestick charts, monthly returns heatmap
            - **📉 Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
            - **⚠️ Risk Analysis**: VaR, volatility metrics, beta, drawdown analysis
            - **📈 Advanced Risk Charts**: Volatility cone, rolling metrics, correlation analysis
            - **🔮 AI Predictions**: Prophet & XGBoost models with confidence intervals
            - **📋 Statistics**: Comprehensive return, risk, and volume statistics
            
            ### ✨ Supports ALL BSE & NSE Stocks
            
            | Exchange | Suffix | Examples |
            |----------|--------|----------|
            | **NSE** | `.NS` | RELIANCE.NS, TCS.NS, ZOMATO.NS, PAYTM.NS |
            | **BSE** | `.BO` | RELIANCE.BO, TCS.BO, TATASTEEL.BO |
            
            > **Tip:** Just enter the stock name (e.g., `RELIANCE`) and we'll add the suffix automatically!
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Example Stocks
            
            **Large Cap:**
            - RELIANCE, TCS, HDFCBANK
            - INFY, ICICIBANK, SBIN
            
            **IT Sector:**
            - TCS, INFY, WIPRO
            - HCLTECH, TECHM
            
            **Banks:**
            - HDFCBANK, ICICIBANK
            - SBIN, KOTAKBANK
            
            **New Age Tech:**
            - ZOMATO, PAYTM
            - NYKAA, POLICYBZR
            
            **PSU:**
            - IRCTC, COALINDIA
            - ONGC, NTPC
            
            **Auto:**
            - TATAMOTORS, MARUTI
            - BAJAJ-AUTO, M&M
            """)
        
        st.divider()
        st.markdown(DISCLAIMER)


if __name__ == "__main__":
    main()

