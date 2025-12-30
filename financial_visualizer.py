"""
FINANCIAL VISUALIZER - CFA PSM CONCEPTS
Advanced financial data visualization
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class FinancialVisualizer:
    """
    Implements financial visualization techniques from CFA PSM:
    - Efficient frontier plots
    - Risk-return tradeoff visualization
    - Time series analysis charts
    - Correlation heatmaps
    - Performance attribution
    """
    
    def __init__(self):
        """Initialize visualizer with Plotly templates"""
        self.template = "plotly_white"
        
    def plot_efficient_frontier(self, simulation_results, optimal_portfolio):
        """
        Plot Monte Carlo simulation results as efficient frontier
        (CFA: Visualizing risk-return tradeoff)
        """
        # Extract data
        returns = simulation_results['return'].values
        volatilities = simulation_results['volatility'].values
        sharpes = simulation_results['sharpe'].values
        
        # Create scatter plot of all simulations
        fig = go.Figure()
        
        # All simulated portfolios
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers',
            marker=dict(
                size=8,
                color=sharpes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.6
            ),
            name='Random Portfolios',
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "Sharpe: %{marker.color:.3f}<extra></extra>"
            )
        ))
        
        # Optimal portfolio
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility']],
            y=[optimal_portfolio['return']],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Optimal Portfolio',
            hovertemplate=(
                "Optimal Portfolio<br>"
                "Volatility: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "Sharpe: %{marker.color:.3f}<extra></extra>"
            )
        ))
        
        # Layout
        fig.update_layout(
            title='Efficient Frontier - Monte Carlo Simulation',
            xaxis_title='Annual Volatility (Risk)',
            yaxis_title='Annual Expected Return',
            hovermode='closest',
            template=self.template,
            height=600,
            showlegend=True
        )
        
        # Add Sharpe ratio contours (advanced)
        sharpe_levels = [0.5, 1.0, 1.5, 2.0]
        for sr in sharpe_levels:
            # Points where return = rf + sr * volatility
            x_vals = np.linspace(min(volatilities), max(volatilities), 50)
            y_vals = 0.03 + sr * x_vals  # Assuming 3% risk-free rate
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(dash='dash', width=1, color='gray'),
                name=f'Sharpe = {sr}',
                showlegend=False,
                opacity=0.5
            ))
        
        return fig
    
    def plot_asset_allocation(self, weights, asset_names):
        """Plot portfolio allocation as interactive pie/donut chart"""
        # Create DataFrame for plotting
        allocation_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights
        }).sort_values('Weight', ascending=False)
        
        # Create donut chart
        fig = px.pie(
            allocation_df,
            values='Weight',
            names='Asset',
            hole=0.4,
            title='Portfolio Asset Allocation'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>"
        )
        
        fig.update_layout(
            template=self.template,
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_cumulative_returns(self, returns_data, benchmark_returns=None):
        """
        Plot cumulative returns over time
        (CFA: Performance visualization)
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_data).cumprod()
        
        # Create figure
        fig = go.Figure()
        
        # Plot each asset
        colors = px.colors.qualitative.Set3
        for i, column in enumerate(cumulative_returns.columns):
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                mode='lines',
                name=column,
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=(
                    "Date: %{x}<br>"
                    "Asset: " + column + "<br>"
                    "Cumulative Return: %{y:.3f}<extra></extra>"
                )
            ))
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative,
                mode='lines',
                name='Benchmark',
                line=dict(width=3, color='black', dash='dash'),
                hovertemplate=(
                    "Date: %{x}<br>"
                    "Benchmark Cumulative Return: %{y:.3f}<extra></extra>"
                )
            ))
        
        # Layout
        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (1 = starting value)',
            hovermode='x unified',
            template=self.template,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add horizontal line at 1.0 (starting value)
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
        
        return fig
    
    def plot_correlation_heatmap(self, returns_data):
        """Plot correlation matrix as heatmap (CFA: Diversification analysis)"""
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate=(
                "Asset X: %{y}<br>"
                "Asset Y: %{x}<br>"
                "Correlation: %{z:.3f}<extra></extra>"
            )
        ))
        
        # Layout
        fig.update_layout(
            title='Asset Correlation Matrix',
            xaxis_title='Assets',
            yaxis_title='Assets',
            template=self.template,
            height=500,
            width=600
        )
        
        return fig
    
    def plot_risk_metrics_dashboard(self, returns_series, window=252):
        """
        Create dashboard of rolling risk metrics
        (CFA: Time-varying risk analysis)
        """
        # Calculate rolling metrics
        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252)
        rolling_var = returns_series.rolling(window=window).apply(
            lambda x: np.percentile(x, 5)
        )
        
        # Calculate drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Annualized Volatility', 
                           'Rolling 95% Value at Risk',
                           'Portfolio Drawdown'),
            vertical_spacing=0.1,
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='blue', width=2),
                hovertemplate="Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Rolling VaR
        fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode='lines',
                name='95% VaR',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)',
                hovertemplate="Date: %{x}<br>VaR: %{y:.2%}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)',
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>"
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Risk Metrics Dashboard',
            height=800,
            template=self.template,
            showlegend=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(tickformat=".0%", row=3, col=1)
        
        # Add zero line for drawdown
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        return fig
    
    def plot_performance_attribution(self, portfolio_returns, benchmark_returns, 
                                   asset_returns, weights):
        """
        Plot performance attribution (CFA: Understanding return sources)
        """
        # Calculate active return
        active_return = portfolio_returns - benchmark_returns
        
        # Calculate contribution of each asset (simplified)
        n_assets = len(weights)
        asset_contributions = pd.DataFrame(index=portfolio_returns.index)
        
        for i in range(n_assets):
            asset_contributions[f'Asset_{i}'] = weights[i] * asset_returns.iloc[:, i]
        
        # Calculate cumulative contributions
        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()
        cum_active = (1 + active_return).cumprod()
        
        # Create figure
        fig = go.Figure()
        
        # Portfolio vs Benchmark
        fig.add_trace(go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio.values,
            mode='lines',
            name='Portfolio',
            line=dict(width=3, color='blue'),
            hovertemplate="Portfolio: %{y:.3f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=cum_benchmark.index,
            y=cum_benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(width=3, color='gray', dash='dash'),
            hovertemplate="Benchmark: %{y:.3f}<extra></extra>"
        ))
        
        # Active return (difference)
        fig.add_trace(go.Scatter(
            x=cum_active.index,
            y=cum_active.values,
            mode='lines',
            name='Active Return',
            line=dict(width=2, color='green'),
            hovertemplate="Active: %{y:.3f}<extra></extra>",
            yaxis="y2"
        ))
        
        # Layout with dual y-axes
        fig.update_layout(
            title='Performance Attribution: Portfolio vs Benchmark',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            yaxis2=dict(
                title="Active Return (Right)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig


# ============================================================================
# EXAMPLE USAGE & DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CFA PSM FINANCIAL VISUALIZER DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252*2, freq='B')
    
    # Sample asset returns
    asset_returns = pd.DataFrame({
        'US_Stocks': np.random.normal(0.0007, 0.015, len(dates)),
        'Intl_Stocks': np.random.normal(0.0006, 0.017, len(dates)),
        'Bonds': np.random.normal(0.0002, 0.005, len(dates)),
        'Gold': np.random.normal(0.0003, 0.012, len(dates))
    }, index=dates)
    
    # Sample portfolio returns (60% US, 20% Intl, 15% Bonds, 5% Gold)
    portfolio_weights = [0.60, 0.20, 0.15, 0.05]
    portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
    
    # Benchmark returns (S&P 500 like)
    benchmark_returns = pd.Series(
        np.random.normal(0.0006, 0.014, len(dates)),
        index=dates
    )
    
    # Create visualizer
    visualizer = FinancialVisualizer()
    
    print("\n📊 Generated sample data for visualization examples")
    print(f"• {len(dates)} trading days")
    print(f"• {len(asset_returns.columns)} assets")
    print(f"• Portfolio weights: {portfolio_weights}")
    
    print("\n🎨 Available visualizations (CFA PSM concepts):")
    print("1. Efficient frontier plots")
    print("2. Asset allocation charts")  
    print("3. Cumulative returns visualization")
    print("4. Correlation heatmaps")
    print("5. Risk metrics dashboards")
    print("6. Performance attribution charts")
    
    # Example: Create correlation heatmap
    print("\n🔥 Example: Correlation Heatmap")
    corr_fig = visualizer.plot_correlation_heatmap(asset_returns)
    print("✓ Correlation heatmap created (can save as HTML or display)")
    
    # Example: Create cumulative returns chart
    print("\n📈 Example: Cumulative Returns Chart")
    cum_fig = visualizer.plot_cumulative_returns(asset_returns, benchmark_returns)
    print("✓ Cumulative returns chart created")
    
    # Example: Create asset allocation chart
    print("\n🎯 Example: Asset Allocation Chart")
    alloc_fig = visualizer.plot_asset_allocation(portfolio_weights, asset_returns.columns.tolist())
    print("✓ Asset allocation chart created")
    
    print("\n" + "=" * 60)
    print("USAGE IN PRACTICE:")
    print("=" * 60)
    print("""
# In your portfolio analysis script:
from financial_visualizer import FinancialVisualizer

# Initialize visualizer
viz = FinancialVisualizer()

# Create efficient frontier plot
fig = viz.plot_efficient_frontier(simulation_results, optimal_portfolio)
fig.write_html("efficient_frontier.html")  # Save as interactive HTML
fig.show()  # Display in Jupyter/Streamlit

# Create risk dashboard
risk_fig = viz.plot_risk_metrics_dashboard(portfolio_returns)
risk_fig.write_image("risk_dashboard.png")  # Save as PNG
    """)
    
    print("\n" + "=" * 60)
    print("CFA Visualization Concepts Demonstrated:")
    print("• Risk-return tradeoff visualization")
    print("• Performance attribution charts")
    print("• Time series analysis techniques")
    print("• Interactive financial dashboards")
    print("• Professional financial reporting visuals")
    print("=" * 60)