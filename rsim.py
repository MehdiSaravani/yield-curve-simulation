"""
Yield Curve Simulation using PCA + VAR(1) Model

This module implements a yield curve simulation framework that:
1. Extracts latent factors from historical yield data using PCA
2. Models factor dynamics with a VAR(1) process  
3. Simulates future yield curve paths via Monte Carlo
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from typing import Optional
import warnings

def load_yield_data(filepath: str, sheet_name: int | str = 0,
                    min_date: Optional[str] = None,
                    scale: float = 1) -> pd.DataFrame:
    """
    Load and validate yield curve data from an Excel file, with optional scaling.

    This function performs comprehensive data validation including:
    - Date column identification and conversion
    - Integer column name enforcement (maturities in months)
    - Monthly frequency validation (no gaps, no duplicates)
    - Missing data handling and numeric type enforcement
    - Optional scaling of yield values via the `scale` parameter

    Parameters
    ----------
    filepath : str
        Path to the Excel file containing yield data.
    sheet_name : int or str, optional
        Sheet name or index to read from the Excel file. Default is 0 (first sheet).
    min_date : str, optional
        Minimum date (inclusive) to filter the data by. Example: "2000-01-01".
    scale : float, optional
        Multiplier applied to all numeric yield values after conversion to float and
        before sanity checks. Use this when the source data is not in decimal format.
        For example, set `scale=0.01` if the Excel values are in percent (5 becomes 0.05).
        Default is 1 (no scaling).

    Returns
    -------
    pd.DataFrame
        Validated yield curve data indexed by `Date` (DatetimeIndex) with integer column
        names representing maturities in months. Yield values are scaled by `scale`.

    Raises
    ------
    ValueError
        If data validation fails for any of the specified requirements.
    FileNotFoundError
        If the specified file doesn't exist.

    Notes
    -----
    - Scaling is applied right after converting the DataFrame to `float` and BEFORE the
      unreasonable-value range check. This ensures the range check operates on the
      scaled values. The default range check warns for values outside [-0.10, 0.50]
      i.e., -10% to 50% in decimal terms.
    - Column names are coerced to integers by extracting digits, assuming maturities in
      months. Columns that cannot be coerced will raise an error.
    """
    try:
        # Read Excel file
        df = pd.read_excel(filepath, sheet_name=sheet_name)

    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Drop rows with NaN values
    df = df.dropna()

    # Identify date column (should be first column or named 'Date'/'date')
    date_column = None
    potential_date_cols = [df.columns[0]]  # First column by default

    # Also check for columns explicitly named as date-related
    for col in df.columns:
        if isinstance(col, str) and col.lower() in ['date', 'dates', 'time', 'period']:
            potential_date_cols.append(col)

    # Find the actual date column by trying to parse each candidate
    for col in potential_date_cols:
        try:
            pd.to_datetime(df[col], errors='raise')
            date_column = col
            break
        except (ValueError, TypeError):
            continue

    if date_column is None:
        raise ValueError(
            "No valid date column found. Expected first column to contain dates "
            "or a column named 'Date'/'date'"
        )

    print(f"Using '{date_column}' as date column")

    # Convert date column to datetime and set as index
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.index.name = 'Date'
    except Exception as e:
        raise ValueError(f"Failed to convert date column to datetime: {e}")

    # filter dates from min_date if provided
    if min_date is not None:
        df = df[df.index >= pd.to_datetime(min_date)]

    # Remove any remaining columns that aren't numeric yield data
    numeric_cols = []
    non_numeric_cols = []

    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        df = df[numeric_cols]

    if df.empty:
        raise ValueError("No numeric columns found in the data")

    # Enforce integer column names (representing maturities in months)
    new_columns = {}
    for col in df.columns:
        try:
            # Try to convert to integer
            if isinstance(col, str):
                # Remove any non-digit characters and convert
                clean_col = ''.join(filter(str.isdigit, str(col)))
                if clean_col:
                    new_columns[col] = int(clean_col)
                else:
                    raise ValueError(f"Cannot convert column '{col}' to integer")
            else:
                new_columns[col] = int(col)
        except (ValueError, TypeError):
            raise ValueError(
                f"Column '{col}' cannot be converted to integer. "
                "All yield columns must have integer names representing maturity in months."
            )

    df.rename(columns=new_columns, inplace=True)

    # Sort columns by maturity (ascending)
    df = df.reindex(sorted(df.columns), axis=1)

    print(f"Final column names (maturities in months): {list(df.columns)}")

    # Drop rows with any NaN values in yield data
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)

    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NaN values")

    if df.empty:
        raise ValueError("No valid data remaining after removing NaN values")

    # Validate monthly frequency and no gaps
    df_sorted = df.sort_index()
    date_diff = df_sorted.index.to_series().diff()[1:]  # Skip first NaT

    # Check for monthly separation (allow 28-31 days)
    monthly_days = date_diff.dt.days

    # Check if all differences are approximately monthly (28-32 days to account for different month lengths)
    non_monthly = monthly_days[(monthly_days < 28) | (monthly_days > 32)]

    if len(non_monthly) > 0:
        problem_dates = df_sorted.index[monthly_days[(monthly_days < 28) | (monthly_days > 32)].index]
        raise ValueError(
            f"Data is not monthly separated. Found {len(non_monthly)} non-monthly gaps. "
            f"Problem dates: {problem_dates.tolist()[:5]}..."  # Show first 5 problem dates
        )

    # Check for same-month duplicates
    monthly_periods = df_sorted.index.to_period('M')
    duplicated_months = monthly_periods.duplicated()

    if duplicated_months.any():
        duplicate_dates = df_sorted.index[duplicated_months]
        raise ValueError(
            f"Found duplicate months in data: {duplicate_dates.tolist()[:5]}..."
        )

    # Check for month gaps
    expected_periods = pd.period_range(
        start=monthly_periods.min(),
        end=monthly_periods.max(),
        freq='M'
    )

    missing_months = expected_periods.difference(monthly_periods)

    if len(missing_months) > 0:
        raise ValueError(
            f"Found {len(missing_months)} missing months in the time series. "
            f"Missing periods: {missing_months.tolist()[:5]}..."
        )

    # Final validation - ensure all values are numeric
    try:
        df = df.astype(float)
        # Apply scaling to yield values
        df = df * scale
    except Exception as e:
        raise ValueError(f"Failed to convert yield data to numeric values: {e}")

    # Check for reasonable yield values (between -10% and 50%)
    unreasonable_values = (df < -0.10) | (df > 0.50)
    if unreasonable_values.any().any():
        warnings.warn(
            "Some yield values appear unreasonable (outside -10% to 50% range). "
            "Please verify data is in decimal format (e.g., 0.05 for 5%)",
            UserWarning
        )

    print(f"Data validation successful!")
    print(f"Final data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Maturities: {list(df.columns)} months")

    return df


class YieldCurveSimulator:
    """
    Yield curve simulator using PCA factor extraction and VAR(1) dynamics.
    
    The model decomposes yield curves into principal components, fits a VAR(1)
    to factor dynamics, then simulates future paths by evolving factors and
    reconstructing yield curves.
    """
    
    def __init__(self, n_factors: int = 3):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        n_factors : int, optional
            Number of principal components to extract. Default is 3.
        """
        self.n_factors = n_factors
        self.pca = None
        self.var_model = None
        self.var_res = None
        self.X = None  # PCA factor scores
        self.Y = None  # Original yield data
        self.is_fitted = False
        
    def fit(self, yield_data: pd.DataFrame):
        """
        Fit the PCA + VAR(1) model to historical yield data.
        
        Parameters
        ----------
        yield_data : pd.DataFrame
            Historical yield curve data. Rows are time periods, 
            columns are maturities.
            
        Returns
        -------
        self : YieldCurveSimulator
            Fitted model instance.
        """
        self.Y = yield_data.values
        self.maturities = yield_data.columns.values
        self.n_observations, self.n_maturities = self.Y.shape
        
        # Extract factors via PCA
        self.pca = PCA(n_components=self.n_factors)
        self.X = self.pca.fit_transform(self.Y)
        
        # Fit VAR(1) to factor dynamics
        self.var_model = VAR(self.X)
        self.var_res = self.var_model.fit(maxlags=1)
        
        self.is_fitted = True
        return
        
    def simulate(self, n_paths: int = 1000, n_steps: int = 120, 
                 sim_start_index: int | list[int] = -1, seed: int = 420) -> list[np.ndarray]:
        """
        Simulate future yield curve paths using Monte Carlo.
        
        Parameters
        ----------
        n_paths : int, optional
            Number of simulation paths. Default is 1000.
        n_steps : int, optional
            Number of time steps per path. Default is 120.
        sim_start_index : int or list[int], optional
            Starting factor values (index from historical data). Default is -1 (last).
        seed : int, optional
            Random seed for reproducibility. Default is 420.
            
        Returns
        -------
        list[np.ndarray]
            [X_sim, Y_sim] where:
            - X_sim: Factor paths (n_paths, n_steps, n_factors)
            - Y_sim: Yield paths (n_paths, n_steps, n_maturities)
            
        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation. Call fit() first.")
        
        if type(sim_start_index) is int:
            sim_start_index = [sim_start_index]
        n_start_index = len(sim_start_index)
        n_total_paths = n_paths * n_start_index

        # Extract VAR(1) parameters
        c = self.var_res.params[0]  # intercept
        Phi = self.var_res.coefs[0]  # coefficient matrix
        Sigma = self.var_res.sigma_u  # shock covariance
        L = self.pca.components_  # loading matrix
        mu = self.pca.mean_  # PCA mean
        
        # Generate all random shocks at once (vectorized)
        rng = np.random.default_rng(seed=seed)
        all_shocks = rng.multivariate_normal(
            mean=np.zeros(self.n_factors),
            cov=Sigma,
            size=(n_total_paths, n_steps - 1)
        )
        
        # Initialize factor simulation array
        X_sim = np.zeros((n_total_paths, n_steps, self.n_factors))
        for i in range(n_start_index):
            x0 = self.X[sim_start_index[i]]
            X_sim[n_paths*i:n_paths*(i+1), 0, :] = x0  # Set initial conditions
        
        # Evolve factors using VAR(1) dynamics (vectorized over paths)
        for t in range(1, n_steps):
            X_sim[:, t, :] = (c[np.newaxis, :] + 
                             X_sim[:, t-1, :] @ Phi.T + 
                             all_shocks[:, t-1, :])
        
        # Reconstruct yield curves (vectorized tensor operation)
        Y_sim = mu[np.newaxis, np.newaxis, :] + X_sim @ L
        
        return [X_sim, Y_sim]
        
    def get_model_summary(self, decimal_places: int = 4) -> dict:
        """
        Get model diagnostics and fit statistics.
        
        Parameters
        ----------
        decimal_places : int, optional
            Decimal places for rounding. Default is 4.
            
        Returns
        -------
        dict
            Model summary with fit statistics and diagnostics.
            
        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary.")
        
        return {
            'n_factors': self.n_factors,
            'n_maturities': self.n_maturities,
            'n_observations': len(self.X),
            'explained_variance_ratio': np.round(self.pca.explained_variance_ratio_, decimal_places),
            'cumulative_explained_variance': round(float(np.sum(self.pca.explained_variance_ratio_)), decimal_places),
        }

    def plot_factor_analysis(self):
        """
        Combined plot showing factor statistics and loadings:
        - Top row: 4 subplots with factor histograms and time series
        - Bottom row: Factor loadings plots including line plot across maturities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")
        
        X = self.X
        L = self.pca.components_  # Loading matrix (n_factors x n_maturities)
        factor_names = ['Level', 'Slope', 'Curvature'] if self.n_factors == 3 else [f'Factor {i+1}' for i in range(self.n_factors)]
        maturities = self.maturities

        # Create figure with 2 rows: top for factor stats, bottom for loadings
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        fig.suptitle('PCA Factor Analysis: Statistics and Loadings', fontsize=18, fontweight='bold')
        
        # ======================== TOP ROW: FACTOR STATISTICS ========================
        
        # Plot histograms for each factor in the first 3 subplots of top row
        for i in range(min(3, self.n_factors)):
            ax = fig.add_subplot(gs[0, i])
            
            # Create histogram
            n_bins = min(30, max(10, len(X) // 20))  # Adaptive bin count
            ax.hist(X[:, i], bins=n_bins, density=True, alpha=0.7, 
                color=f'C{i}', edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(X[:, i])
            std_val = np.std(X[:, i])
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'μ: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'+σ: {mean_val + std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'-σ: {mean_val - std_val:.3f}')
            
            ax.set_xlabel('Factor Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{factor_names[i]} Distribution\n({self.pca.explained_variance_ratio_[i]:.1%} variance)', 
                     fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
        # Time series plot in the rightmost subplot of top row
        ax_time = fig.add_subplot(gs[0, 3])
    
        for i in range(self.n_factors):
            ax_time.plot(X[:, i], label=factor_names[i],
                     linewidth=1.5, alpha=0.8)
    
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Factor Score')
        ax_time.set_title('Factor Evolution Over Time')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
        # ======================== BOTTOM ROW: LOADINGS ========================

        ax_lines = fig.add_subplot(gs[1, :])
    
        for i in range(self.n_factors):
            factor_name = factor_names[i]
            ax_lines.plot(maturities/12, L[i, :], label=factor_name,
                     linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    
        ax_lines.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_lines.set_title('Factor Loadings vs Maturity')
        ax_lines.set_xlabel('Maturity (Years)')
        ax_lines.set_ylabel('Loading')
        ax_lines.legend()
        ax_lines.grid(True, alpha=0.3)

        plt.show()

    def plot_yield_timeseries(
            self,
            Y_sim: np.ndarray,
            maturities: list[int],
            time_axis: Optional[np.ndarray] = None,
            fan_chart: bool = False,
            overlay_history: bool = False,
            history_steps: int = 24
    ):
        """
        Plot simulated yield statistics for multiple maturities.

        Parameters
        ----------
        Y_sim : np.ndarray
            Simulated yields (n_paths, n_steps, n_maturities)
        maturities : list[int]
            List of maturities in months
        time_axis : np.ndarray, optional
            Custom x-axis
        fan_chart : bool
            If True, plots fan chart instead of discrete bands
        overlay_history : bool
            If True, overlays last historical observations
        history_steps : int
            Number of historical steps to overlay
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting.")

        for m in maturities:
            if m not in self.maturities:
                raise ValueError(
                    f"Maturity {m} not found. Available: {list(self.maturities)}"
                )

        n_mats = len(maturities)
        n_cols = 2
        n_rows = int(np.ceil(n_mats / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(14, 4 * n_rows),
            sharex=False
        )

        axes = np.array(axes).reshape(-1)

        for idx, maturity in enumerate(maturities):

            ax = axes[idx]

            m_idx = np.where(self.maturities == maturity)[0][0]
            Y_m = Y_sim[:, :, m_idx]  # (n_paths, n_steps)

            mean_path = np.mean(Y_m, axis=0)

            quantiles = {
                q: np.quantile(Y_m, q, axis=0)
                for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
            }

            n_steps = Y_m.shape[1]
            x = time_axis if time_axis is not None else np.arange(n_steps)

            if fan_chart:

                fan_levels = [
                    (0.05, 0.95),
                    (0.10, 0.90),
                    (0.25, 0.75)
                ]

                for i, (q_low, q_high) in enumerate(fan_levels):
                    ax.fill_between(
                        x,
                        quantiles[q_low],
                        quantiles[q_high],
                        alpha=0.15 + 0.1 * i
                    )

                ax.plot(x, mean_path, linewidth=2.5, label="Mean")

            else:

                ax.fill_between(
                    x, quantiles[0.05], quantiles[0.25],
                    alpha=0.3, label="5–25%"
                )

                ax.fill_between(
                    x, quantiles[0.75], quantiles[0.95],
                    alpha=0.3, label="75–95%"
                )

                ax.plot(x, mean_path, linewidth=2.5, label="Mean")

            if overlay_history:
                hist = self.Y[-history_steps:, m_idx]

                hist_x = np.arange(-history_steps, 0)

                ax.plot(
                    hist_x,
                    hist,
                    linestyle='--',
                    linewidth=2,
                    label="Historical"
                )

                ax.axvline(0, linestyle=":", linewidth=1)

            ax.set_title(f"{maturity / 12:.1f}Y Maturity")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Yield")
            ax.grid(alpha=0.3)
            ax.legend()

        # hide unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Simulated Yield Distribution by Maturity", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_yield_distribution_comparison(
            self,
            Y_sim: np.ndarray,
            maturities: list[int],
            horizon_step: int = -1,
            bins: int = 20
    ):
        """
        Compare historical yield distribution to simulated distribution at given horizon.
        """

        n = len(maturities)
        n_cols = 2
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i, m in enumerate(maturities):
            ax = axes[i]
            m_idx = np.where(self.maturities == m)[0][0]

            hist_data = self.Y[:, m_idx]
            sim_data = Y_sim[:, horizon_step, m_idx]

            ax.hist(hist_data, bins=bins, density=True, alpha=0.5, label="Historical")
            ax.hist(sim_data, bins=bins, density=True, alpha=0.5, label="Simulated")

            ax.set_title(f"{m / 12:.1f}Y Distribution")
            ax.legend()
            ax.grid(alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Historical vs Simulated Yield Distribution")
        plt.tight_layout()
        plt.show()

    def plot_delta_y_distribution(
            self,
            Y_sim: np.ndarray,
            maturities: list[int],
            bins: int = 20
    ):
        """
        Compare 1-step yield change distribution.
        """

        n = len(maturities)
        n_cols = 2
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        hist_dY = np.diff(self.Y, axis=0)
        sim_dY = np.diff(Y_sim, axis=1)

        for i, m in enumerate(maturities):
            ax = axes[i]
            m_idx = np.where(self.maturities == m)[0][0]

            ax.hist(hist_dY[:, m_idx], bins=bins, density=True, alpha=0.5, label="Historical")
            ax.hist(sim_dY[:, :, m_idx].flatten(), bins=bins, density=True, alpha=0.5, label="Simulated")

            ax.set_title(f"{m / 12:.1f}Y ΔY Distribution")
            ax.legend()
            ax.grid(alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Historical vs Simulated Yield Changes")
        plt.tight_layout()
        plt.show()

    def plot_slope_curvature_diagnostics(
            self,
            Y_sim: np.ndarray,
            short_maturity: int,
            mid_maturity: int,
            long_maturity: int
    ):
        """
        Plot slope and curvature time series + distribution.
        """

        s = np.where(self.maturities == short_maturity)[0][0]
        m = np.where(self.maturities == mid_maturity)[0][0]
        l = np.where(self.maturities == long_maturity)[0][0]

        # Historical
        hist_slope = self.Y[:, l] - self.Y[:, s]
        hist_curv = 2 * self.Y[:, m] - self.Y[:, l] - self.Y[:, s]

        # Simulated (flatten over paths)
        sim_slope = Y_sim[:, :, l] - Y_sim[:, :, s]
        sim_curv = 2 * Y_sim[:, :, m] - Y_sim[:, :, l] - Y_sim[:, :, s]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        axes[0, 0].hist(hist_slope, bins=20, density=True)
        axes[0, 0].set_title("Historical Slope Dist")

        axes[0, 1].hist(sim_slope.flatten(), bins=20, density=True)
        axes[0, 1].set_title("Simulated Slope Dist")

        axes[1, 0].hist(hist_curv, bins=20, density=True)
        axes[1, 0].set_title("Historical Curvature Dist")

        axes[1, 1].hist(sim_curv.flatten(), bins=20, density=True)
        axes[1, 1].set_title("Simulated Curvature Dist")

        plt.tight_layout()
        plt.show()

    def plot_correlation_comparison(
            self,
            Y_sim: np.ndarray,
            horizon_step: int = -1
    ):
        """
        Compare historical vs simulated yield correlation matrices.
        """

        hist_corr = np.corrcoef(self.Y.T)
        sim_corr = np.corrcoef(Y_sim[:, horizon_step, :].T)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes[0].imshow(hist_corr, aspect='auto')
        axes[0].set_title("Historical Correlation")

        im2 = axes[1].imshow(sim_corr, aspect='auto')
        axes[1].set_title("Simulated Correlation")

        for ax in axes:
            ax.set_xticks(range(len(self.maturities)))
            ax.set_yticks(range(len(self.maturities)))
            ax.set_xticklabels(self.maturities, rotation=90)
            ax.set_yticklabels(self.maturities)

        plt.colorbar(im2, ax=axes.ravel().tolist())
        plt.show()