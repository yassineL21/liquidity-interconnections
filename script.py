"""
Liquidity Interconnection Analysis for Financial Markets
Author: Youssef Karim El Alaoui
Purpose: Generate real statistical results for research paper
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import math

warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LiquidityAnalysis:
    """Complete liquidity analysis framework"""
    
    def __init__(self, tickers, start_date='1996-01-01', end_date='2024-12-31'):
        """
        Initialize with market tickers
        
        Parameters:
        -----------
        tickers : dict
            Dictionary of {name: ticker_symbol}
        start_date : str
            Start date for data
        end_date : str
            End date for data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.liquidity_measures = {}
        
    def download_data(self):
        """Download price and volume data from Yahoo Finance"""
        print("Downloading data from Yahoo Finance...")
        
        for name, ticker in self.tickers.items():
            print(f"  Downloading {name} ({ticker})...")
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, 
                               progress=False)
                if len(df) > 0:
                    self.data[name] = df
                    print(f"    ✓ {len(df)} observations")
                else:
                    print(f"    ✗ No data available")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print(f"\nSuccessfully downloaded data for {len(self.data)} indices")
        return self
    
    def calculate_returns(self, df):
        """Calculate log returns"""
        return np.log(df['Close'] / df['Close'].shift(1))
    
    def calculate_amihud(self, df, window=20):
        """
        Calculate Amihud (2002) illiquidity measure
        
        ILLIQ = (1/N) * sum(|Return| / Volume)
        
        Higher values = lower liquidity
        """
        returns = self.calculate_returns(df)
        volume = df['Volume']
        
        # Daily illiquidity
        daily_illiq = np.abs(returns) / (volume + 1)  # +1 to avoid division by zero
        
        # Monthly average (rolling window)
        monthly_illiq = daily_illiq.rolling(window=window, min_periods=10).mean()
        
        # Log transform and scale (multiply by 10^6 for readability)
        log_illiq = np.log(monthly_illiq * 1e6)
        
        return log_illiq
    
    def calculate_roll(self, df, window=20):
        """
        Calculate Roll (1984) estimator of bid-ask spread
        
        Roll = 2 * sqrt(-Cov(R_t, R_{t-1}))
        
        Only defined when covariance is negative
        """
        returns = self.calculate_returns(df)
        
        # Calculate rolling covariance
        def rolling_cov(x):
            if len(x) < 2:
                return np.nan
            return np.cov(x[:-1], x[1:])[0, 1]
        
        roll_cov = returns.rolling(window=window).apply(rolling_cov, raw=False)
        
        # Roll estimator (set to 0 when covariance is positive)
        roll = 2 * np.sqrt(np.maximum(-roll_cov, 0))
        
        return roll * 100  # Convert to percentage
    
    def calculate_corwin_schultz(self, df):
        """
        Calculate Corwin-Schultz (2012) bid-ask spread estimator
        
        Uses high-low price ranges over two consecutive days
        """
        high = df['High'].values
        low = df['Low'].values
        
        # Initialize result array
        cs = np.zeros(len(df))
        
        # Calculate for each day (starting from day 1, since we need t-1)
        for i in range(1, len(df)):
            # Beta: sum of squared log ranges over two days
            beta = (np.log(high[i] / low[i]) ** 2 + 
                    np.log(high[i-1] / low[i-1]) ** 2)
            
            # Gamma: squared log range of max high to min low over two days
            max_high = max(high[i], high[i-1])
            min_low = min(low[i], low[i-1])
            gamma = np.log(max_high / min_low) ** 2
            
            # Alpha calculation
            sqrt_2 = np.sqrt(2)
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * sqrt_2) - \
                    np.sqrt(gamma / (3 - 2 * sqrt_2))
            
            # CS spread
            if alpha > 0:
                cs[i] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            else:
                cs[i] = 0
        
        # Convert to Series with proper index
        cs_series = pd.Series(cs * 100, index=df.index)  # Convert to percentage
        
        return cs_series
    
    def compute_all_liquidity_measures(self):
        """Calculate all three liquidity measures for all indices"""
        print("\nCalculating liquidity measures...")
        
        for name, df in self.data.items():
            print(f"  Processing {name}...")
            
            measures = pd.DataFrame(index=df.index)
            measures['Amihud'] = self.calculate_amihud(df)
            measures['Roll'] = self.calculate_roll(df)
            measures['CS'] = self.calculate_corwin_schultz(df)
            
            # Drop NaN values
            measures = measures.dropna()
            
            self.liquidity_measures[name] = measures
            print(f"    ✓ {len(measures)} valid observations")
        
        return self
    
    def get_monthly_data(self, measure='Amihud'):
        """
        Convert daily liquidity measures to monthly averages
        
        Parameters:
        -----------
        measure : str
            Which measure to use ('Amihud', 'Roll', or 'CS')
        """
        monthly_data = pd.DataFrame()
        
        for name, df in self.liquidity_measures.items():
            monthly = df[measure].resample('M').mean()
            monthly_data[name] = monthly
        
        # Align all series (common dates only)
        monthly_data = monthly_data.dropna()
        
        return monthly_data
    
    def descriptive_statistics(self, measure='Amihud'):
        """Generate descriptive statistics table"""
        print(f"\n{'='*60}")
        print(f"Descriptive Statistics: {measure} Measure")
        print(f"{'='*60}")
        
        stats_table = pd.DataFrame()
        
        for name, df in self.liquidity_measures.items():
            series = df[measure].dropna()
            
            stats_table.loc[name, 'Mean'] = series.mean()
            stats_table.loc[name, 'Std'] = series.std()
            stats_table.loc[name, 'Min'] = series.min()
            stats_table.loc[name, 'Max'] = series.max()
            stats_table.loc[name, 'Skewness'] = series.skew()
            stats_table.loc[name, 'Kurtosis'] = series.kurtosis()
            stats_table.loc[name, 'N'] = len(series)
        
        print(stats_table.round(3))
        print()
        
        return stats_table
    
    def correlation_analysis(self, measure='Amihud', periods=None):
        """
        Calculate correlation matrices for different periods
        
        Parameters:
        -----------
        measure : str
            Liquidity measure to analyze
        periods : dict or None
            Dictionary of {period_name: (start_date, end_date)}
        """
        if periods is None:
            # Define default crisis and normal periods
            periods = {
                'Full Sample': ('1996-01-01', '2024-12-31'),
                'Normal (2003-06)': ('2003-01-01', '2006-12-31'),
                'GFC (2007-09)': ('2007-01-01', '2009-12-31'),
                'Normal (2013-19)': ('2013-01-01', '2019-12-31'),
                'COVID-19 (2020)': ('2020-01-01', '2020-12-31')
            }
        
        monthly = self.get_monthly_data(measure)
        
        print(f"\n{'='*60}")
        print(f"Correlation Analysis: {measure} Measure")
        print(f"{'='*60}\n")
        
        correlation_summary = pd.DataFrame()
        
        for period_name, (start, end) in periods.items():
            period_data = monthly.loc[start:end]
            
            if len(period_data) < 2:
                print(f"{period_name}: Insufficient data")
                continue
            
            corr_matrix = period_data.corr()
            
            # Calculate average correlations by region
            us_indices = ['DJIA', 'SP500', 'NASDAQ']
            eur_indices = ['FTSE', 'DAX', 'CAC']
            
            # US-US correlations
            us_corr = []
            for i in range(len(us_indices)):
                for j in range(i+1, len(us_indices)):
                    if us_indices[i] in corr_matrix.columns and us_indices[j] in corr_matrix.columns:
                        us_corr.append(corr_matrix.loc[us_indices[i], us_indices[j]])
            
            # EUR-EUR correlations
            eur_corr = []
            for i in range(len(eur_indices)):
                for j in range(i+1, len(eur_indices)):
                    if eur_indices[i] in corr_matrix.columns and eur_indices[j] in corr_matrix.columns:
                        eur_corr.append(corr_matrix.loc[eur_indices[i], eur_indices[j]])
            
            # US-EUR correlations
            us_eur_corr = []
            for us in us_indices:
                for eur in eur_indices:
                    if us in corr_matrix.columns and eur in corr_matrix.columns:
                        us_eur_corr.append(corr_matrix.loc[us, eur])
            
            # Overall average
            upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            correlation_summary.loc[period_name, 'US-US'] = np.mean(us_corr) if us_corr else np.nan
            correlation_summary.loc[period_name, 'EUR-EUR'] = np.mean(eur_corr) if eur_corr else np.nan
            correlation_summary.loc[period_name, 'US-EUR'] = np.mean(us_eur_corr) if us_eur_corr else np.nan
            correlation_summary.loc[period_name, 'Overall'] = np.mean(upper_tri)
            correlation_summary.loc[period_name, 'N'] = len(period_data)
            
            print(f"{period_name} (N={len(period_data)}):")
            print(corr_matrix.round(3))
            print()
        
        print("\nSummary Table:")
        print(correlation_summary.round(3))
        print()
        
        return correlation_summary
    
    def test_stationarity(self, measure='Amihud'):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Null hypothesis: Unit root (non-stationary)
        """
        monthly = self.get_monthly_data(measure)
        
        print(f"\n{'='*60}")
        print(f"Stationarity Tests (ADF): {measure} Measure")
        print(f"{'='*60}\n")
        
        results = pd.DataFrame()
        
        for col in monthly.columns:
            series = monthly[col].dropna()
            
            # ADF test
            adf_result = adfuller(series, autolag='AIC')
            
            results.loc[col, 'ADF Statistic'] = adf_result[0]
            results.loc[col, 'p-value'] = adf_result[1]
            results.loc[col, 'Lags Used'] = adf_result[2]
            results.loc[col, 'Stationary'] = 'Yes' if adf_result[1] < 0.05 else 'No'
        
        print(results.round(4))
        print()
        
        return results
    
    def estimate_var(self, measure='Amihud', maxlags=12, ic='bic'):
        """
        Estimate VAR model and select optimal lag order
        
        Parameters:
        -----------
        measure : str
            Liquidity measure to use
        maxlags : int
            Maximum number of lags to consider
        ic : str
            Information criterion ('aic', 'bic', or 'hqic')
        """
        monthly = self.get_monthly_data(measure)
    
        print(f"\n{'='*60}")
        print(f"VAR Model Estimation: {measure} Measure")
        print(f"{'='*60}\n")

        # Fit VAR model
        model = VAR(monthly)

        # Select optimal lag order
        print("Lag Order Selection:")
        lag_order = model.select_order(maxlags=maxlags)
        print(lag_order.summary())
        print()

        # Get optimal lag order based on the selected criterion
        # Access the attribute directly from the LagOrderResults object
        if ic == 'aic':
            optimal_lag = lag_order.aic
        elif ic == 'bic':
            optimal_lag = lag_order.bic
        elif ic == 'hqic':
            optimal_lag = lag_order.hqic
        else:
            print(f"Warning: Unknown criterion '{ic}'. Using BIC instead.")
            optimal_lag = lag_order.bic

        print(f"Selected lag order ({ic.upper()}): {optimal_lag}\n")

        # Fit model with optimal lag
        var_result = model.fit(optimal_lag)

        print("VAR Model Summary:")
        print(var_result.summary())
        print()

        return var_result, monthly
    
    def granger_causality_analysis(self, var_result, monthly_data, maxlag=4):
        """
        Perform pairwise Granger causality tests
        
        Tests if one variable helps predict another
        """
        print(f"\n{'='*60}")
        print(f"Granger Causality Tests (maxlag={maxlag})")
        print(f"{'='*60}\n")
        
        variables = monthly_data.columns.tolist()
        n_vars = len(variables)
        
        # Create results matrix
        causality_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)
        
        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if i == j:
                    causality_matrix.loc[cause, effect] = np.nan
                    continue
                
                try:
                    # Granger causality test
                    test_result = grangercausalitytests(
                        monthly_data[[effect, cause]], 
                        maxlag=maxlag, 
                        verbose=False
                    )
                    
                    # Get p-value for F-test at selected lag
                    p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                    min_p = min(p_values)
                    
                    causality_matrix.loc[cause, effect] = min_p
                    
                except Exception as e:
                    causality_matrix.loc[cause, effect] = np.nan
        
        print("P-values (H0: cause does NOT Granger-cause effect):")
        print(causality_matrix.round(4))
        print()
        
        # Create significance matrix
        sig_matrix = causality_matrix.copy()
        sig_matrix = sig_matrix.applymap(lambda x: '***' if x < 0.01 else 
                                         ('**' if x < 0.05 else 
                                          ('*' if x < 0.10 else 'NS')))
        
        print("Significance levels (* p<0.10, ** p<0.05, *** p<0.01):")
        print(sig_matrix)
        print()
        
        return causality_matrix, sig_matrix
    
    def impulse_response_analysis(self, var_result, periods=10):
        """
        Calculate and plot Generalized Impulse Response Functions
        
        Shows how shocks propagate across markets
        """
        print(f"\n{'='*60}")
        print(f"Impulse Response Analysis")
        print(f"{'='*60}\n")
        
        # Calculate IRFs
        irf = var_result.irf(periods)
        
        # Plot IRFs
        fig = irf.plot(orth=False, figsize=(16, 12))
        plt.suptitle('Generalized Impulse Response Functions', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig('irf_all.png', dpi=300, bbox_inches='tight')
        print("Saved: irf_all.png")
        
        # Calculate cumulative responses
        cumulative_irf = irf.cum_effects
        
        print("\nCumulative responses (10 periods):")
        print(pd.DataFrame(cumulative_irf[-1], 
                          columns=var_result.names,
                          index=var_result.names).round(4))
        
        return irf

    
    def plot_granger_causality_network(self, significance_matrix, pvalue_matrix=None):
        """
        Creates and plots a directed network graph from Granger causality results.
        """
        
        # Create a directed graph
        G = nx.DiGraph()
        markets = significance_matrix.index.tolist()
        G.add_nodes_from(markets)
        
        # --- DEBUG: Print what we're reading from the matrix ---
        print(f"\n{'='*60}")
        print("DEBUG: Reading significance matrix for FTSE")
        print(f"{'='*60}")
        
        # --- Build the network ---
        significant_edges = []
        for cause in markets:
            for effect in markets:
                if cause == effect:
                    continue
                
                # Read: Does 'cause' Granger-cause 'effect'?
                # significance_matrix[effect, cause] = sig for "cause -> effect"
                sig = significance_matrix.loc[effect, cause]
                
                # DEBUG for FTSE specifically
                if cause == 'FTSE':
                    print(f"  Checking FTSE -> {effect}: significance = '{sig}'")
                
                if sig in ['**', '***']:
                    edge_attrs = {'sig_level': sig}
                    if pvalue_matrix is not None:
                        pval = pvalue_matrix.loc[effect, cause]
                        edge_attrs['pvalue'] = pval
                        edge_attrs['weight'] = 1.0 + 2.0 * (-np.log10(pval) / 10)
                    
                    G.add_edge(cause, effect, **edge_attrs)
                    significant_edges.append((cause, effect))
        
        # --- DEBUG: Verify the graph structure ---
        print(f"\nGraph has {G.number_of_edges()} edges total")
        print(f"Edges found: {significant_edges}")
        
        # Calculate ACTUAL out-degree from the graph
        out_degrees = dict(G.out_degree())
        print(f"\nActual out-degrees from graph:")
        for market in markets:
            print(f"  {market}: {out_degrees.get(market, 0)} outgoing edges")
        
        # FTSE should have out-degree of 5 (causing all 5 other markets)
        if 'FTSE' in out_degrees:
            print(f"\n✓ FTSE causes {out_degrees['FTSE']} other markets (should be 5)")
        
        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a layout that spreads nodes nicely
        pos = nx.spring_layout(G, seed=42, k=3.0)  # Increased k for more spacing
        
        # Color nodes by region
        us_markets = ['DJIA', 'SP500', 'NASDAQ']
        node_colors = ['#1f77b4' if node in us_markets else '#ff7f0e' for node in G.nodes()]
        
        # CRITICAL FIX: Calculate node sizes from ACTUAL out-degree
        # FTSE (causing 5 others) should be largest, others smaller
        if G.number_of_edges() > 0:
            # Find the maximum out-degree to scale all nodes
            max_out = max(out_degrees.values()) if out_degrees else 1
            print(f"\nMaximum out-degree: {max_out}")
            
            # Scale: base size + bonus for influence
            node_sizes = []
            for node in G.nodes():
                out_deg = out_degrees.get(node, 0)
                # Make influential nodes MUCH larger
                size = 800 + 1200 * 1.5 # (out_deg / max_out)
                node_sizes.append(size)
                print(f"  {node}: out_degree={out_deg}, size={size:.0f}")
        else:
            node_sizes = [800] * len(markets)
        
        # Prepare edge widths (optional, based on p-values)
        if G.number_of_edges() > 0 and pvalue_matrix is not None:
            edge_widths = []
            for u, v in G.edges():
                pval = G[u][v].get('pvalue', 0.05)
                width = 1.0 + 3.0 * (1.0 - min(pval/0.05, 1.0))
                edge_widths.append(width)
        else:
            edge_widths = [2.0] * G.number_of_edges()
        
                # --- Drawing in correct order ---
        
        # 1. Draw nodes first (since arrows will point TO them)
        node_sizes = [2000] * len(G.nodes())
        nx.draw_networkx_nodes(G, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.85,
                               edgecolors='black',
                               linewidths=1.5,
                               ax=ax)
        
        # 2. Draw labels
        nx.draw_networkx_labels(G, pos,
                                font_size=11,
                                font_weight='bold',
                                ax=ax)
        
        # 3. Draw arrows with OFFSET (MANUAL CONTROL)
        if G.number_of_edges() > 0:            
            # CONTROL: Adjust this value to change arrow offset
            # Higher = more space between arrow tip and node
            ARROW_OFFSET = 0.06  # Start with 0.05, adjust as needed
            
            for (u, v) in G.edges():
                # Get node positions
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Calculate direction vector
                dx = x2 - x1
                dy = y2 - y1
                
                # Calculate vector length
                length = math.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Normalize the vector
                    dx /= length
                    dy /= length
                    
                    # Calculate start and end points with offsets
                    # Start from source node + offset
                    start_x = x1 + dx * ARROW_OFFSET#* 0.8  # Small offset from source
                    start_y = y1 + dy * ARROW_OFFSET# * 0.8
                    
                    # End at target node - offset (arrow stops before reaching node)
                    end_x = x2 - dx * ARROW_OFFSET
                    end_y = y2 - dy * ARROW_OFFSET
                    
                    # Get edge styling
                    if pvalue_matrix is not None and 'weight' in G[u][v]:
                        edge_width = G[u][v]['weight']
                    else:
                        edge_width = 2.0
                    
                    # Create the arrow with control points for slight curvature
                    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                            arrowstyle='-|>',
                                            connectionstyle='arc3,rad=0.1',
                                            color='#2e2e2e',
                                            linewidth=edge_width,
                                            alpha=0.9,
                                            mutation_scale=25)  # Controls arrowhead size
                    
                    ax.add_patch(arrow)
        
        print(f"✓ Arrows drawn with offset: {ARROW_OFFSET}")
        
        # --- Finalize plot ---
        ax.set_title('Granger Causality Network of Market Liquidity (Significant at 5% Level)',
                     fontsize=14, pad=20)
        ax.axis('off')
        
        # Add legend
        import matplotlib.patches as mpatches
        us_patch = mpatches.Patch(color='#1f77b4', label='US Markets')
        eur_patch = mpatches.Patch(color='#ff7f0e', label='European Markets')
        ax.legend(handles=[us_patch, eur_patch], loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig('granger_causality_network_corrected.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Corrected network diagram saved as 'granger_causality_network_corrected.png'")
        
        # --- Network statistics ---
        print(f"\n{'='*60}")
        print("FINAL NETWORK STATISTICS")
        print(f"{'='*60}")
        
        print(f"\nMost influential markets (by outgoing causal links):")
        sorted_markets = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        for market, degree in sorted_markets:
            print(f"  {market}: Causes {degree} other market(s)")
        
        return G
    
    def plot_liquidity_evolution(self, measure='Amihud'):
        """
        Create publication-quality time series plot
        """
        monthly = self.get_monthly_data(measure)
        
        # Define crisis periods
        crises = [
            ('2000-03-01', '2002-10-01', 'Dot-com', 'lightcoral'),
            ('2007-10-01', '2009-03-01', 'GFC', 'lightblue'),
            ('2010-05-01', '2012-07-01', 'Europe', 'lightgreen'),
            ('2020-02-01', '2020-12-01', 'COVID-19', 'lightyellow')
        ]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each series
        for col in monthly.columns:
            ax.plot(monthly.index, monthly[col], label=col, linewidth=1.5, alpha=0.8)
        
        # Shade crisis periods
        for start, end, name, color in crises:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                      alpha=0.2, color=color, label=name)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{measure} Illiquidity', fontsize=12)
        ax.set_title(f'Evolution of {measure} Liquidity Measure (1996-2024)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{measure.lower()}_evolution.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {measure.lower()}_evolution.png")
        
        return fig

class RobustnessAnalysis:
    """
    Comprehensive robustness testing for liquidity interconnection results.
    """
    
    def __init__(self, analyzer):
        """
        Initialize with a fitted LiquidityAnalysis object.
        
        Parameters:
        -----------
        analyzer : LiquidityAnalysis
            Already fitted analyzer with computed liquidity measures.
        """
        self.analyzer = analyzer
        self.results = {}
        
    def test_alternative_measures(self):
        """
        Replicate full analysis using Roll and CS measures.
        
        Returns:
        --------
        dict with keys 'Roll' and 'CS' containing:
            - correlation_summary: Correlation tables by period
            - causality_matrices: Granger causality p-values and significance
            - network_stats: Node out-degrees and hub rankings
        """
        print(f"\n{'='*60}")
        print("ROBUSTNESS CHECK: Alternative Liquidity Measures")
        print(f"{'='*60}")
        
        measures_results = {}
        
        for measure in ['Roll', 'CS']:
            print(f"\nAnalyzing {measure} measure...")
            
            # Get monthly data for this measure
            monthly_data = self.analyzer.get_monthly_data(measure)
            
            # Estimate VAR
            var_result, _ = self.analyzer.estimate_var(measure, maxlags=12, ic='bic')
            
            # Granger causality
            causality_pvals, causality_sig = self.analyzer.granger_causality_analysis(
                var_result, monthly_data, maxlag=4
            )
            
            # Calculate network statistics
            G = self._build_causality_network(causality_sig)
            out_degrees = dict(G.out_degree())
            
            # Store results
            measures_results[measure] = {
                'causality_pvals': causality_pvals,
                'causality_sig': causality_sig,
                'out_degrees': out_degrees,
                'top_hub': max(out_degrees, key=out_degrees.get) if out_degrees else None
            }
            
            print(f"  Top hub for {measure}: {measures_results[measure]['top_hub']} "
                  f"(out-degree: {out_degrees.get(measures_results[measure]['top_hub'], 0)})")
        
        self.results['alternative_measures'] = measures_results
        return measures_results
    
    def test_lag_sensitivity(self, measure='Amihud', lags_to_test=[2, 3]):
        """
        Test sensitivity to different VAR lag orders.
        
        Parameters:
        -----------
        measure : str
            Liquidity measure to use
        lags_to_test : list
            List of lag orders to test (besides BIC-selected)
        """
        print(f"\n{'='*60}")
        print("ROBUSTNESS CHECK: Lag Order Sensitivity")
        print(f"{'='*60}")
        
        monthly_data = self.analyzer.get_monthly_data(measure)
        lag_results = {}
        
        # Get baseline (BIC-selected) model
        var_baseline, _ = self.analyzer.estimate_var(measure, maxlags=12, ic='bic')
        
        for p in lags_to_test:
            print(f"\nTesting VAR({p})...")
            
            # Fit VAR with specific lag
            model = VAR(monthly_data)
            var_result = model.fit(p)
            
            # Granger causality
            causality_pvals, causality_sig = self.analyzer.granger_causality_analysis(
                var_result, monthly_data, maxlag=4
            )
            
            # Compare with baseline
            G = self._build_causality_network(causality_sig)
            out_degrees = dict(G.out_degree())
            
            lag_results[p] = {
                'out_degrees': out_degrees,
                'top_hub': max(out_degrees, key=out_degrees.get) if out_degrees else None,
                'num_edges': G.number_of_edges()
            }
            
            print(f"  Top hub at lag {p}: {lag_results[p]['top_hub']} "
                  f"(out-degree: {out_degrees.get(lag_results[p]['top_hub'], 0)})")
        
        self.results['lag_sensitivity'] = lag_results
        return lag_results
    
    def test_subsample_stability(self, measure='Amihud', exclude_periods=None):
        """
        Test results on subsamples excluding crisis periods.
        
        Parameters:
        -----------
        measure : str
            Liquidity measure to use
        exclude_periods : list of tuples
            [(start1, end1), (start2, end2)] periods to exclude
        """
        if exclude_periods is None:
            exclude_periods = [('2007-01-01', '2009-12-31'),  # GFC
                               ('2020-01-01', '2020-12-31')]  # COVID-19
        
        print(f"\n{'='*60}")
        print("ROBUSTNESS CHECK: Subsample Stability (Excluding Crises)")
        print(f"{'='*60}")
        
        # Get full monthly data
        full_monthly = self.analyzer.get_monthly_data(measure)
        
        # Create subsample (exclude crisis periods)
        subsample = full_monthly.copy()
        for start, end in exclude_periods:
            mask = ~((subsample.index >= start) & (subsample.index <= end))
            subsample = subsample[mask]
        
        print(f"Original sample: {len(full_monthly)} months")
        print(f"Subsample (excl. crises): {len(subsample)} months")
        
        if len(subsample) < 24:  # Minimum for VAR
            print("Warning: Subsample too small for analysis")
            return None
        
        # Analyze subsample
        model = VAR(subsample)
        var_result = model.fit(1)  # Use p=1 for consistency
        
        causality_pvals, causality_sig = self.analyzer.granger_causality_analysis(
            var_result, subsample, maxlag=4
        )
        
        G = self._build_causality_network(causality_sig)
        out_degrees = dict(G.out_degree())
        
        subsample_results = {
            'causality_sig': causality_sig,
            'out_degrees': out_degrees,
            'top_hub': max(out_degrees, key=out_degrees.get) if out_degrees else None,
            'sample_size': len(subsample)
        }
        
        print(f"\nSubsample results:")
        print(f"  Top hub: {subsample_results['top_hub']}")
        print(f"  FTSE out-degree: {out_degrees.get('FTSE', 0)}")
        
        self.results['subsample'] = subsample_results
        return subsample_results
    
    def test_stationarity_transformations(self, measure='Amihud'):
        """
        Test results on first-differenced (stationary) series.
        """
        print(f"\n{'='*60}")
        print("ROBUSTNESS CHECK: First-Differenced Series")
        print(f"{'='*60}")
        
        monthly_data = self.analyzer.get_monthly_data(measure)
        
        # First-difference the series
        diff_data = monthly_data.diff().dropna()
        
        print(f"Original series length: {len(monthly_data)}")
        print(f"First-differenced length: {len(diff_data)}")
        
        if len(diff_data) < 24:
            print("Warning: Differenced series too short")
            return None
        
        # Test stationarity of differenced series
        print("\nADF tests on first-differenced series:")
        for col in diff_data.columns:
            adf_result = adfuller(diff_data[col].dropna(), autolag='AIC')
            stationary = 'Yes' if adf_result[1] < 0.05 else 'No'
            print(f"  {col}: p-value={adf_result[1]:.4f}, Stationary={stationary}")
        
        # Analyze differenced data
        model = VAR(diff_data)
        var_result = model.fit(1)
        
        causality_pvals, causality_sig = self.analyzer.granger_causality_analysis(
            var_result, diff_data, maxlag=4
        )
        
        G = self._build_causality_network(causality_sig)
        out_degrees = dict(G.out_degree())
        
        diff_results = {
            'causality_sig': causality_sig,
            'out_degrees': out_degrees,
            'top_hub': max(out_degrees, key=out_degrees.get) if out_degrees else None
        }
        
        print(f"\nFirst-differenced results:")
        print(f"  Top hub: {diff_results['top_hub']}")
        print(f"  FTSE out-degree: {out_degrees.get('FTSE', 0)}")
        
        self.results['first_differenced'] = diff_results
        return diff_results
    
    def _build_causality_network(self, significance_matrix):
        """
        Helper: Build network from significance matrix.
        """
        G = nx.DiGraph()
        markets = significance_matrix.index.tolist()
        G.add_nodes_from(markets)
        
        for cause in markets:
            for effect in markets:
                if cause == effect:
                    continue
                sig = significance_matrix.loc[effect, cause]
                if sig in ['**', '***']:
                    G.add_edge(cause, effect)
        
        return G
    
    def summarize_results(self):
        """
        Generate a comprehensive summary of all robustness checks.
        """
        print(f"\n{'='*60}")
        print("ROBUSTNESS CHECKS SUMMARY")
        print(f"{'='*60}")
        
        summary = {}
        
        # 1. Alternative measures
        if 'alternative_measures' in self.results:
            print("\n1. Alternative Liquidity Measures:")
            for measure, res in self.results['alternative_measures'].items():
                hub = res['top_hub']
                out_deg = res['out_degrees'].get(hub, 0)
                print(f"   {measure}: Top hub = {hub} (out-degree: {out_deg})")
                summary[f'{measure}_hub'] = hub
                summary[f'{measure}_out_deg'] = out_deg
        
        # 2. Lag sensitivity
        if 'lag_sensitivity' in self.results:
            print("\n2. Lag Order Sensitivity:")
            for lag, res in self.results['lag_sensitivity'].items():
                print(f"   VAR({lag}): Top hub = {res['top_hub']} "
                      f"(edges: {res['num_edges']})")
                summary[f'lag_{lag}_hub'] = res['top_hub']
        
        # 3. Subsample stability
        if 'subsample' in self.results:
            res = self.results['subsample']
            print(f"\n3. Subsample (excl. crises):")
            print(f"   Top hub = {res['top_hub']}, "
                  f"FTSE out-degree = {res['out_degrees'].get('FTSE', 0)}")
            summary['subsample_hub'] = res['top_hub']
        
        # 4. First-differenced
        if 'first_differenced' in self.results:
            res = self.results['first_differenced']
            print(f"\n4. First-Differenced Series:")
            print(f"   Top hub = {res['top_hub']}, "
                  f"FTSE out-degree = {res['out_degrees'].get('FTSE', 0)}")
            summary['diff_hub'] = res['top_hub']
        
        # Calculate consistency score
        hubs = [v for k, v in summary.items() if 'hub' in k]
        if hubs:
            from collections import Counter
            hub_counts = Counter(hubs)
            most_common = hub_counts.most_common(1)[0]
            consistency = most_common[1] / len(hubs) * 100
            print(f"\nOverall Consistency: {consistency:.1f}% "
                  f"({most_common[0]} is top hub in {most_common[1]}/{len(hubs)} tests)")
        
        return summary

    def plot_robustness_chord_diagram(self, figsize=(12, 10)):
        """
        Creates a chord diagram visualizing consistency across robustness tests.
        Shows which markets maintain hub status across different specifications.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Wedge, FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap
        
        print(f"\n{'='*60}")
        print("GENERATING ROBUSTNESS CONSISTENCY VISUALIZATION")
        print(f"{'='*60}")
        
        # Collect data from all robustness tests
        consistency_data = {}
        markets = ['DJIA', 'SP500', 'NASDAQ', 'FTSE', 'DAX', 'CAC']
        
        # Initialize tracking
        for market in markets:
            consistency_data[market] = {
                'as_hub': 0,      # Times identified as primary hub
                'total_tests': 0,  # Total tests considered
                'out_degrees': []  # Out-degree values across tests
            }
        
        # Process each robustness test
        test_names = []
        
        if 'alternative_measures' in self.results:
            test_names.append('Roll Measure')
            test_names.append('CS Measure')
            for measure, res in self.results['alternative_measures'].items():
                hub = res.get('top_hub')
                out_degrees = res.get('out_degrees', {})
                if hub:
                    consistency_data[hub]['as_hub'] += 1
                for market in markets:
                    consistency_data[market]['total_tests'] += 1
                    consistency_data[market]['out_degrees'].append(
                        out_degrees.get(market, 0)
                    )
        
        if 'lag_sensitivity' in self.results:
            for lag, res in self.results['lag_sensitivity'].items():
                test_names.append(f'VAR({lag})')
                hub = res.get('top_hub')
                out_degrees = res.get('out_degrees', {})
                if hub:
                    consistency_data[hub]['as_hub'] += 1
                for market in markets:
                    consistency_data[market]['total_tests'] += 1
                    consistency_data[market]['out_degrees'].append(
                        out_degrees.get(market, 0)
                    )
        
        if 'subsample' in self.results:
            test_names.append('Excl. Crises')
            res = self.results['subsample']
            hub = res.get('top_hub')
            out_degrees = res.get('out_degrees', {})
            if hub:
                consistency_data[hub]['as_hub'] += 1
            for market in markets:
                consistency_data[market]['total_tests'] += 1
                consistency_data[market]['out_degrees'].append(
                    out_degrees.get(market, 0)
                )
        
        if 'first_differenced' in self.results:
            test_names.append('First-Diff.')
            res = self.results['first_differenced']
            hub = res.get('top_hub')
            out_degrees = res.get('out_degrees', {})
            if hub:
                consistency_data[hub]['as_hub'] += 1
            for market in markets:
                consistency_data[market]['total_tests'] += 1
                consistency_data[market]['out_degrees'].append(
                    out_degrees.get(market, 0)
                )
        
        # Calculate consistency metrics
        consistency_metrics = {}
        for market in markets:
            data = consistency_data[market]
            if data['total_tests'] > 0:
                hub_ratio = data['as_hub'] / len(test_names) if test_names else 0
                avg_out_degree = np.mean(data['out_degrees']) if data['out_degrees'] else 0
                std_out_degree = np.std(data['out_degrees']) if len(data['out_degrees']) > 1 else 0
                
                consistency_metrics[market] = {
                    'hub_ratio': hub_ratio,
                    'avg_out_degree': avg_out_degree,
                    'std_out_degree': std_out_degree,
                    'consistency_score': hub_ratio * (1 - min(std_out_degree/3, 1))
                }
        
        # Create the visualization
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Color scheme
        us_colors = ['#1f77b4', '#1f77b4', '#1f77b4']  # Blue for US markets
        eu_colors = ['#ff7f0e', '#ff7f0e', '#ff7f0e']  # Orange for EU markets
        
        # Create a custom colormap for consistency scores
        cmap = LinearSegmentedColormap.from_list('consistency', ['#d62728', '#ff7f0e', '#2ca02c'])
        
        # Prepare data for the chord diagram
        n_markets = len(markets)
        angles = np.linspace(0, 2 * np.pi, n_markets, endpoint=False)
        radii = [consistency_metrics[m]['avg_out_degree'] * 0.8 + 1 for m in markets]
        
        # Draw the outer ring (market segments)
        for i, market in enumerate(markets):
            # Determine color based on region
            color = us_colors[i] if i < 3 else eu_colors[i-3]
            
            # Wedge for each market
            width = 2 * np.pi / n_markets
            start_angle = angles[i] - width/2
            end_angle = angles[i] + width/2
            
            # Draw the wedge
            wedge = Wedge((0, 0), radii[i], np.degrees(start_angle), 
                         np.degrees(end_angle), width=0.3, 
                         color=color, alpha=0.7)
            ax.add_patch(wedge)
            
            # Add market label
            label_angle = angles[i]
            label_radius = radii[i] + 0.5
            ax.text(label_angle, label_radius, market, 
                    ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Add consistency score as inner text
            if market in consistency_metrics:
                score = consistency_metrics[market]['consistency_score']
                ax.text(label_angle, radii[i] - 0.3, f'{score:.2f}', 
                        ha='center', va='center', fontsize=9)
        
        # Draw connecting chords based on consistency relationships
        # Connect markets that are both frequently hubs
        for i in range(n_markets):
            for j in range(i+1, n_markets):
                m1, m2 = markets[i], markets[j]
                if (m1 in consistency_metrics and m2 in consistency_metrics and
                    consistency_metrics[m1]['hub_ratio'] > 0.3 and 
                    consistency_metrics[m2]['hub_ratio'] > 0.3):
                    
                    # Calculate connection strength
                    strength = min(consistency_metrics[m1]['hub_ratio'], 
                                  consistency_metrics[m2]['hub_ratio'])
                    
                    # Draw chord
                    theta1, theta2 = angles[i], angles[j]
                    r1, r2 = radii[i] * 0.7, radii[j] * 0.7
                    
                    # Bezier-like curve for visual appeal
                    t = np.linspace(0, 1, 50)
                    curve_r = r1 * (1-t) + r2 * t
                    curve_theta = theta1 * (1-t) + theta2 * t
                    
                    # Convert to Cartesian for plotting
                    x = curve_r * np.cos(curve_theta)
                    y = curve_r * np.sin(curve_theta)
                    
                    ax.plot(x, y, color='gray', alpha=0.3 * strength, 
                           linewidth=2 * strength, zorder=1)
        
        # Set plot limits and remove polar grid
        ax.set_ylim(0, max(radii) + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Add title and legend
        ax.set_title('Robustness Analysis: Market Consistency Across Specifications', 
                     fontsize=14, pad=20, fontweight='bold')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='US Markets',
                   markerfacecolor='#1f77b4', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='European Markets',
                   markerfacecolor='#ff7f0e', markersize=10),
            Line2D([0], [0], color='gray', lw=2, label='Frequent Co-Hub Relationship',
                   alpha=0.5)
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1.05, 1), fontsize=10)
        
        # Add explanation text box
        explanation_text = (
            "Visualization Explanation:\n"
            "• Wedge size = Average out-degree across tests\n"
            "• Inner number = Consistency score (0-1)\n"
            "• Gray connections = Markets frequently hubs together\n"
            "• Higher/More connected = More robust hub"
        )
        
        ax.text(1.3, 0.5, explanation_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('robustness_chord_diagram.png', dpi=300, bbox_inches='tight')
        print("✓ Robustness chord diagram saved as 'robustness_chord_diagram.png'")
        
        # Print summary statistics
        print("\nConsistency Summary:")
        print("-" * 40)
        for market in markets:
            if market in consistency_metrics:
                metrics = consistency_metrics[market]
                print(f"{market}: Hub in {consistency_data[market]['as_hub']}/{len(test_names)} tests | "
                      f"Avg out-degree: {metrics['avg_out_degree']:.2f} ± {metrics['std_out_degree']:.2f} | "
                      f"Score: {metrics['consistency_score']:.3f}")
        
        return consistency_metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("LIQUIDITY INTERCONNECTION ANALYSIS")
    print("Financial Markets Research")
    print("="*60)
    
    # Define market indices
    tickers = {
        'DJIA': '^DJI',      # Dow Jones Industrial Average
        'SP500': '^GSPC',    # S&P 500
        'NASDAQ': '^IXIC',   # NASDAQ Composite
        'FTSE': '^FTSE',     # FTSE 100
        'DAX': '^GDAXI',     # DAX
        'CAC': '^FCHI'       # CAC 40
    }
    
    # Initialize analysis
    analyzer = LiquidityAnalysis(tickers, start_date='1996-01-01', end_date='2024-12-09')
    
    # Step 1: Download data
    analyzer.download_data()
    
    # Step 2: Calculate liquidity measures
    analyzer.compute_all_liquidity_measures()
    
    # Step 3: Descriptive statistics
    stats_amihud = analyzer.descriptive_statistics('Amihud')
    stats_roll = analyzer.descriptive_statistics('Roll')
    stats_cs = analyzer.descriptive_statistics('CS')
    
    # Step 4: Correlation analysis
    corr_results = analyzer.correlation_analysis('Amihud')
    
    # Step 5: Stationarity tests
    stationarity = analyzer.test_stationarity('Amihud')
    
    # Step 6: VAR estimation
    var_result, monthly_data = analyzer.estimate_var('Amihud', maxlags=12, ic='bic')
    
    # Step 7: Granger causality
    causality_pvals, causality_sig = analyzer.granger_causality_analysis(
        var_result, monthly_data, maxlag=4
    )
    
    # Step 7.5: Plot the causality network
    network_graph = analyzer.plot_granger_causality_network(causality_sig, causality_pvals)
    
    # Step 8: Impulse response analysis
    irf = analyzer.impulse_response_analysis(var_result, periods=12)
    
    # Step 9: Visualizations
    analyzer.plot_liquidity_evolution('Amihud')



    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS")
    print("="*60)
    
    # Initialize robustness analysis
    robustness = RobustnessAnalysis(analyzer)
    
    # 1. Test alternative liquidity measures
    robustness.test_alternative_measures()
    
    # 2. Test lag sensitivity
    robustness.test_lag_sensitivity(measure='Amihud', lags_to_test=[2, 3])
    
    # 3. Test subsample stability (exclude crises)
    robustness.test_subsample_stability(measure='Amihud')
    
    # 4. Test first-differenced series
    robustness.test_stationarity_transformations(measure='Amihud')
    
    # Generate summary
    robustness_summary = robustness.summarize_results()

    # Generate the robustness visualization
    consistency_metrics = robustness.plot_robustness_chord_diagram(figsize=(14, 12))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated outputs:")
    print("  - Descriptive statistics tables")
    print("  - Correlation matrices")
    print("  - VAR model estimates")
    print("  - Granger causality tests")
    print("  - Impulse response functions")
    print("  - Time series plots")
    print("\nAll results ready for your paper!")
