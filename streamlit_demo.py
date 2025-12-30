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


st.header("📊 CFA PSM Portfolio Analysis")

# Import your CFA modules
try:
    from portfolio_optimizer import PortfolioOptimizer
    from risk_analyzer import RiskAnalyzer
    from financial_visualizer import FinancialVisualizer
    
    st.success("✅ CFA Modules Loaded Successfully!")
    
    # Demo of Portfolio Optimizer
    with st.expander("🎯 Portfolio Optimizer Demo"):
        st.write("""
        **Monte Carlo Simulation for Asset Allocation**
        This module implements CFA PSM concepts for portfolio optimization.
        """)
        
        if st.button("Run Monte Carlo Simulation (Sample)"):
            # Create sample data
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=252, freq='B')
            sample_returns = pd.DataFrame({
                'AAPL': np.random.normal(0.0005, 0.02, 252),
                'GOOGL': np.random.normal(0.0004, 0.018, 252),
                'MSFT': np.random.normal(0.0006, 0.015, 252)
            }, index=dates)
            
            # Initialize and run optimizer
            optimizer = PortfolioOptimizer(sample_returns)
            results_df, optimal = optimizer.monte_carlo_simulation(n_simulations=1000)
            
            st.write(f"🏆 **Optimal Portfolio Found:**")
            st.write(f"- Sharpe Ratio: {optimal['sharpe']:.3f}")
            st.write(f"- Expected Return: {optimal['return']:.2%}")
            st.write(f"- Expected Volatility: {optimal['volatility']:.2%}")
    
    # Demo of Risk Analyzer
    with st.expander("⚠️ Risk Analyzer Demo"):
        st.write("""
        **Value at Risk (VaR) and Risk Metrics**
        Implements CFA PSM risk analysis techniques.
        """)
        
        if st.button("Calculate Risk Metrics (Sample)"):
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            portfolio_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
            analyzer = RiskAnalyzer(portfolio_returns)
            
            var_historical = analyzer.historical_var()
            es_historical = analyzer.expected_shortfall(method='historical')
            
            st.write(f"📉 **1-Day 95% VaR:** ${-var_historical*100:.2f} (on $100 portfolio)")
            st.write(f"📊 **Expected Shortfall:** ${-es_historical*100:.2f}")
            
    # Demo of Financial Visualizer
    with st.expander("🎨 Financial Visualizer Demo"):
        st.write("""
        **Interactive Financial Charts and Dashboards**
        CFA PSM data visualization techniques.
        """)
        
        if st.button("Generate Sample Charts"):
            import pandas as pd
            import numpy as np
            
            # Create sample data
            dates = pd.date_range('2020-01-01', periods=252, freq='B')
            asset_returns = pd.DataFrame({
                'Stocks': np.random.normal(0.0007, 0.015, 252),
                'Bonds': np.random.normal(0.0002, 0.005, 252),
                'Gold': np.random.normal(0.0003, 0.012, 252)
            }, index=dates)
            
            visualizer = FinancialVisualizer()
            
            # Show correlation heatmap
            st.write("**Correlation Heatmap:**")
            corr_fig = visualizer.plot_correlation_heatmap(asset_returns)
            st.plotly_chart(corr_fig)
            
except ImportError as e:
    st.warning(f"⚠️ CFA modules not fully loaded: {e}")
    st.info("Make sure all CFA files are in the same directory.")
