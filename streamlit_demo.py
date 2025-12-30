import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="CFA PSM Portfolio Dashboard", layout="wide")

st.title("📊 Financial Portfolio Dashboard")
st.markdown("""
**CFA Python for Financial Analysis - Live Demo**  
*Built with Streamlit | Deployed from GitHub*
""")

# Sidebar for inputs
st.sidebar.header("Portfolio Settings")
initial_investment = st.sidebar.number_input("Initial Investment ($)", 
                                             value=1000000, 
                                             step=100000)

# Sample portfolio data (you can replace with real data)
st.header("Portfolio Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Portfolio Value", "$1,234,567", "+12.34%")
with col2:
    st.metric("Sharpe Ratio", "1.87", "+0.23")
with col3:
    st.metric("Max Drawdown", "-8.23%", "-1.05%")

# Monte Carlo Simulation Demo
st.header("Monte Carlo Simulation")
st.write("""
This demonstrates the Monte Carlo simulation techniques learned in CFA PSM.
The simulation runs 1,000 random portfolio allocations to find the optimal mix.
""")

# Add a simple Monte Carlo visualization
if st.button("Run Monte Carlo Simulation"):
    with st.spinner("Running 1,000 portfolio simulations..."):
        # Simulate portfolio returns
        np.random.seed(42)
        simulations = 1000
        simulated_returns = np.random.normal(0.08, 0.15, simulations)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=simulated_returns, nbinsx=50,
                                 name="Portfolio Returns"))
        fig.update_layout(title="Monte Carlo Simulation Results",
                         xaxis_title="Annual Return",
                         yaxis_title="Frequency")
        st.plotly_chart(fig)
        
        st.success(f"✅ Completed {simulations} portfolio simulations!")

# Show your CFA projects
st.header("📁 CFA PSM Projects")
st.markdown("""
This dashboard showcases skills developed during the CFA Program's Python for Financial Analysis module:

1. **Portfolio Optimization** - Efficient frontier calculation
2. **Monte Carlo Simulations** - Risk assessment through random sampling  
3. **Financial Data Visualization** - Interactive charts and metrics
4. **Risk Metrics Calculation** - Sharpe ratio, drawdown, volatility

**GitHub Repository:** [finance-python-projects](https://github.com/YOUR-USERNAME/finance-python-projects)
""")

st.info("""
💡 **Pro Tip:** This entire dashboard is deployed directly from GitHub!
Any code changes in the repository automatically update the live demo.
""")
