"""
RISK ANALYZER - CFA PSM CONCEPTS
Advanced risk metrics and analysis
"""

import numpy as np
import pandas as pd
from scipy import stats

class RiskAnalyzer:
    """
    Implements risk analysis techniques from CFA PSM:
    - Value at Risk (VaR) - Historical, Parametric, Monte Carlo
    - Expected Shortfall (ES)
    - Stress Testing
    - Risk Decomposition
    """
    
    def __init__(self, returns_series, confidence_level=0.95):
        """
        Initialize with returns data
        
        Parameters:
        returns_series: pandas Series of portfolio returns
        confidence_level: VaR/ES confidence level (e.g., 0.95 for 95%)
        """
        self.returns = returns_series
        self.confidence = confidence_level
        self.alpha = 1 - confidence_level
        
    def historical_var(self):
        """
        Historical VaR (CFA non-parametric method)
        Simple percentile-based approach
        """
        var = np.percentile(self.returns, self.alpha * 100)
        return var
    
    def parametric_var(self, distribution='normal'):
        """
        Parametric VaR (CFA parametric method)
        Assumes returns follow a specific distribution
        
        distribution: 'normal' or 't' (Student's t-distribution)
        """
        if distribution == 'normal':
            # Normal distribution assumption
            mu = self.returns.mean()
            sigma = self.returns.std()
            z_score = stats.norm.ppf(self.alpha)
            var = mu + z_score * sigma
            
        elif distribution == 't':
            # Student's t-distribution (fatter tails)
            params = stats.t.fit(self.returns)
            df, loc, scale = params
            t_score = stats.t.ppf(self.alpha, df)
            var = loc + t_score * scale
            
        return var
    
    def monte_carlo_var(self, n_simulations=10000, days=1):
        """
        Monte Carlo VaR (CFA simulation method)
        Simulates future returns based on historical parameters
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Simulate future returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations * days)
        simulated_returns = simulated_returns.reshape(n_simulations, days)
        
        # Calculate portfolio value after 'days'
        initial_value = 100  # Assume $100 initial
        final_values = initial_value * (1 + simulated_returns).prod(axis=1)
        
        # Calculate losses
        losses = initial_value - final_values
        var = np.percentile(losses, self.confidence * 100)
        
        return var
    
    def expected_shortfall(self, method='historical'):
        """
        Expected Shortfall (CFA: average loss beyond VaR)
        Also known as Conditional VaR (CVaR)
        """
        if method == 'historical':
            var = self.historical_var()
            # Average of returns worse than VaR
            tail_returns = self.returns[self.returns <= var]
            es = tail_returns.mean()
            
        elif method == 'parametric':
            var = self.parametric_var()
            # For normal distribution, ES has closed form
            mu = self.returns.mean()
            sigma = self.returns.std()
            z_alpha = stats.norm.ppf(self.alpha)
            es = mu - sigma * (stats.norm.pdf(z_alpha) / self.alpha)
            
        return es
    
    def calculate_max_drawdown(self, prices):
        """
        Maximum Drawdown (CFA: largest peak-to-trough decline)
        """
        # Calculate cumulative returns if input is returns
        if (prices <= 1).all():  # Likely returns
            cumulative = (1 + prices).cumprod()
        else:  # Already prices
            cumulative = prices
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        max_drawdown_duration = (drawdown == max_drawdown).argmax()
        
        return max_drawdown, max_drawdown_duration
    
    def stress_test(self, shock_size=0.10, method='uniform'):
        """
        Stress Testing (CFA: impact of extreme scenarios)
        
        shock_size: Percentage shock to apply
        method: 'uniform' (all assets down) or 'varied' (different shocks)
        """
        current_value = 100  # Assume $100 portfolio
        
        if method == 'uniform':
            # All assets decline by same percentage
            stressed_value = current_value * (1 - shock_size)
            loss = current_value - stressed_value
            
        elif method == 'varied':
            # Different assets have different shocks (more realistic)
            # Simulate varied shocks between 5% and 15%
            shocks = np.random.uniform(0.05, 0.15, 5)  # 5 assets
            avg_shock = shocks.mean()
            stressed_value = current_value * (1 - avg_shock)
            loss = current_value - stressed_value
        
        return loss, stressed_value
    
    def risk_decomposition(self, portfolio_weights, asset_returns, portfolio_return):
        """
        Risk Decomposition (CFA: contribution of each asset to total risk)
        
        Returns marginal contribution to risk (MCTR) for each asset
        """
        n_assets = len(portfolio_weights)
        covariance_matrix = asset_returns.cov() * 252  # Annualized
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(
            np.dot(portfolio_weights.T, np.dot(covariance_matrix, portfolio_weights))
        )
        
        # Calculate MCTR for each asset
        mctr = []
        for i in range(n_assets):
            # Derivative of portfolio vol with respect to weight i
            cov_with_portfolio = np.dot(covariance_matrix[i], portfolio_weights)
            mctr_i = cov_with_portfolio / portfolio_vol
            mctr.append(mctr_i)
        
        # Calculate percentage contribution
        total_mctr = sum(mctr)
        contribution_pct = [m / total_mctr for m in mctr]
        
        return mctr, contribution_pct


# ============================================================================
# EXAMPLE USAGE & DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CFA PSM RISK ANALYZER DEMONSTRATION")
    print("=" * 60)
    
    # Create sample portfolio returns
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    portfolio_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    print(f"\n📊 Sample Data: {n_days} days of portfolio returns")
    print(f"Mean daily return: {portfolio_returns.mean():.4%}")
    print(f"Daily volatility: {portfolio_returns.std():.4%}")
    
    # Initialize risk analyzer
    analyzer = RiskAnalyzer(portfolio_returns, confidence_level=0.95)
    
    # Calculate various VaR measures
    print("\n" + "=" * 60)
    print("VALUE AT RISK (VaR) CALCULATIONS")
    print("=" * 60)
    
    historical_var = analyzer.historical_var()
    parametric_var_normal = analyzer.parametric_var(distribution='normal')
    parametric_var_t = analyzer.parametric_var(distribution='t')
    monte_carlo_var = analyzer.monte_carlo_var(n_simulations=10000)
    
    print(f"\n📉 1-Day 95% VaR (for $100 portfolio):")
    print(f"  • Historical VaR:      ${-historical_var*100:.2f}")
    print(f"  • Parametric (Normal): ${-parametric_var_normal*100:.2f}")
    print(f"  • Parametric (t-dist): ${-parametric_var_t*100:.2f}")
    print(f"  • Monte Carlo VaR:     ${monte_carlo_var:.2f}")
    
    # Calculate Expected Shortfall
    print("\n" + "=" * 60)
    print("EXPECTED SHORTFALL (ES) / CONDITIONAL VaR")
    print("=" * 60)
    
    es_historical = analyzer.expected_shortfall(method='historical')
    es_parametric = analyzer.expected_shortfall(method='parametric')
    
    print(f"\n📊 1-Day 95% Expected Shortfall:")
    print(f"  • Historical ES:      ${-es_historical*100:.2f}")
    print(f"  • Parametric ES:      ${-es_parametric*100:.2f}")
    print(f"  • ES > VaR (as expected): Expected loss is worse than VaR")
    
    # Calculate Maximum Drawdown
    print("\n" + "=" * 60)
    print("MAXIMUM DRAWDOWN ANALYSIS")
    print("=" * 60)
    
    # Create price series from returns
    prices = 100 * (1 + portfolio_returns).cumprod()
    max_dd, dd_duration = analyzer.calculate_max_drawdown(prices)
    
    print(f"\n📈 Maximum Drawdown: {max_dd:.2%}")
    print(f"  • Peak-to-trough decline")
    print(f"  • Occurred around day {dd_duration}")
    
    # Stress Testing
    print("\n" + "=" * 60)
    print("STRESS TESTING SCENARIOS")
    print("=" * 60)
    
    loss_uniform, stressed_uniform = analyzer.stress_test(shock_size=0.15, method='uniform')
    loss_varied, stressed_varied = analyzer.stress_test(shock_size=0.15, method='varied')
    
    print(f"\n💥 15% Market Shock Scenarios (on $100 portfolio):")
    print(f"  • Uniform shock loss:      ${loss_uniform:.2f}")
    print(f"  • Varied shock loss:       ${loss_varied:.2f}")
    print(f"  • Stressed value (uniform): ${stressed_uniform:.2f}")
    
    print("\n" + "=" * 60)
    print("CFA Risk Concepts Demonstrated:")
    print("• Value at Risk (3 methods)")
    print("• Expected Shortfall / Conditional VaR")
    print("• Maximum Drawdown")
    print("• Stress Testing")
    print("• Risk decomposition (available via risk_decomposition method)")
    print("=" * 60)