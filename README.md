# 📊 Finance Python Projects

Collection of Python tools for financial analysis and portfolio management, developed during CFA Program's Python for Financial Analysis (PSM) studies.

## 🎯 What's Here

### 1. Portfolio Optimizer
- Efficient frontier calculation
- Monte Carlo simulations for risk assessment
- Sharpe ratio optimization

### 2. Financial Dashboard  
- Interactive visualizations with Plotly
- Real-time stock data analysis
- Performance tracking tools

### 3. Risk Analysis Tools
- Value at Risk (VaR) calculations
- Correlation analysis
- Drawdown analysis

## 🛠️ Technologies Used
- **Python 3.10+**
- **Pandas & NumPy** (data manipulation)
- **Matplotlib & Plotly** (visualization)
- **yFinance** (market data)
- **Streamlit** (web apps)

## 📈 Projects in Detail

### Portfolio Optimization
```python
# Example: Monte Carlo simulation for optimal asset allocation
def monte_carlo_portfolio(n_simulations=10000):
    # Simulates 10,000 random portfolios
    # Returns optimal weights based on Sharpe ratio
