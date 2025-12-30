"""
PORTFOLIO OPTIMIZER - CFA PSM CONCEPTS
Monte Carlo simulations for optimal asset allocation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Implements portfolio optimization techniques from CFA PSM:
    - Monte Carlo simulations for efficient frontier
    - Sharpe ratio maximization
    - Risk-return tradeoff analysis
    """
    
    def __init__(self, returns_data):
        """
        Initialize with historical returns data
        
        Parameters:
        returns_data: pandas DataFrame with asset returns (daily recommended)
        """
        self.returns = returns_data
        self.n_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
        
    def generate_random_portfolio(self):
        """Generate random weights that sum to 1 (CFA weight generation concept)"""
        weights = np.random.random(self.n_assets)
        weights = weights / weights.sum()
        return weights
    
    def calculate_portfolio_metrics(self, weights, risk_free_rate=0.03):
        """
        Calculate key portfolio metrics (CFA risk/return formulas)
        
        Returns:
        annual_return, annual_volatility, sharpe_ratio
        """
        # Expected annual return (CFA formula: sum(weights * mean_returns) * 252)
        mean_daily_returns = self.returns.mean()
        portfolio_daily_return = np.dot(weights, mean_daily_returns)
        annual_return = portfolio_daily_return * 252
        
        # Portfolio volatility (CFA formula: sqrt(w.T * Cov * w))
        covariance_matrix = self.returns.cov() * 252  # Annualized covariance
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        annual_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (CFA: (return - rf) / volatility)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return annual_return, annual_volatility, sharpe_ratio
    
    def monte_carlo_simulation(self, n_simulations=10000, risk_free_rate=0.03):
        """
        Run Monte Carlo simulation (CFA PSM core concept)
        
        Returns DataFrame with all simulated portfolios
        """
        print(f"Running {n_simulations:,} Monte Carlo simulations...")
        
        results = []
        
        for i in range(n_simulations):
            # Generate random portfolio
            weights = self.generate_random_portfolio()
            
            # Calculate metrics
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, risk_free_rate)
            
            # Store results
            result = {
                'simulation': i,
                'weights': weights.copy(),
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            }
            
            # Add individual asset weights
            for j, asset in enumerate(self.asset_names):
                result[f'weight_{asset}'] = weights[j]
                
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 2000 == 0:
                print(f"  Completed {i + 1:,} simulations...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal portfolio (max Sharpe ratio)
        optimal_idx = results_df['sharpe'].idxmax()
        optimal_portfolio = results_df.loc[optimal_idx].to_dict()
        
        print(f"✓ Simulation complete. Best Sharpe ratio: {optimal_portfolio['sharpe']:.3f}")
        
        return results_df, optimal_portfolio
    
    def get_portfolio_allocation_summary(self, weights):
        """Create human-readable portfolio allocation summary"""
        allocation = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': [f"{w:.1%}" for w in weights],
            'Allocation': weights
        }).sort_values('Allocation', ascending=False)
        
        return allocation
    
    def calculate_diversification_benefit(self, weights):
        """
        Calculate diversification ratio (CFA concept)
        Ratio of weighted avg individual vol / portfolio vol
        Higher ratio = better diversification
        """
        individual_vols = self.returns.std() * np.sqrt(252)
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        _, portfolio_vol, _ = self.calculate_portfolio_metrics(weights)
        
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return diversification_ratio


# ============================================================================
# EXAMPLE USAGE & DEMO (Shows CFA concepts in action)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CFA PSM PORTFOLIO OPTIMIZER DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data (similar to CFA exercises)
    np.random.seed(42)  # For reproducible results
    n_days = 252 * 3  # 3 years of daily data
    
    # Generate synthetic returns for different asset classes
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    
    returns_data = pd.DataFrame({
        'US_Stocks': np.random.normal(0.0007, 0.015, n_days),    # ~18% annual vol
        'Intl_Stocks': np.random.normal(0.0006, 0.017, n_days),  # ~27% annual vol
        'Bonds': np.random.normal(0.0002, 0.005, n_days),        # ~8% annual vol
        'Real_Estate': np.random.normal(0.0005, 0.012, n_days),  # ~19% annual vol
        'Commodities': np.random.normal(0.0004, 0.020, n_days),  # ~32% annual vol
    }, index=dates)
    
    print(f"\n📊 Sample Data: {len(returns_data)} trading days, {len(returns_data.columns)} assets")
    print(f"Asset classes: {', '.join(returns_data.columns)}")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_data)
    
    # Run Monte Carlo simulation
    results_df, optimal = optimizer.monte_carlo_simulation(n_simulations=5000)
    
    print("\n" + "=" * 60)
    print("🏆 OPTIMAL PORTFOLIO FOUND")
    print("=" * 60)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  • Expected Annual Return: {optimal['return']:.2%}")
    print(f"  • Expected Annual Volatility: {optimal['volatility']:.2%}")
    print(f"  • Sharpe Ratio: {optimal['sharpe']:.3f}")
    
    print(f"\n🎯 Asset Allocation:")
    allocation = optimizer.get_portfolio_allocation_summary(optimal['weights'])
    for _, row in allocation.iterrows():
        print(f"  • {row['Asset']}: {row['Weight']}")
    
    # Calculate diversification benefit
    div_ratio = optimizer.calculate_diversification_benefit(optimal['weights'])
    print(f"\n🛡️ Diversification Benefit Ratio: {div_ratio:.2f}")
    print(f"   (Higher = better diversification)")
    
    print("\n" + "=" * 60)
    print("CFA Concepts Demonstrated:")
    print("• Monte Carlo simulation for portfolio optimization")
    print("• Sharpe ratio maximization")
    print("• Risk-return tradeoff analysis")
    print("• Diversification benefit calculation")
    print("=" * 60)