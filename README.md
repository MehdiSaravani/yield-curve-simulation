# Yield Curve Simulation

Tools and notebooks for simulating and analyzing the U.S. Treasury yield curve using historical data, principal component analysis (PCA), and Vector AutoRegression (VAR) models.

## Contents

- Core Python module (`rsim.py`) providing:
  - `load_yield_data`: load, validate, and scale yield data from Excel.
  - PCA utilities for factor extraction.
  - A simulator class that fits PCA + VAR(1) and simulates future curves.
- Jupyter notebook (`rate simulation.ipynb`) demonstrating the full workflow with plots and commentary.

## Requirements

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels openpyxl
```

## Data

- Place `FRB_H15.xlsx` (or another historical yield dataset) in this folder or provide an absolute path when calling the loader.
- Data source: Federal Reserve (https://www.federalreserve.gov/DataDownload/)

## Using `rsim.load_yield_data`

Function signature:

```
load_yield_data(filepath: str, sheet_name: int | str = 0, min_date: str | None = None, scale: float = 1) -> pd.DataFrame
```

What it does (high level):
- Detects and parses a date column, setting a `DatetimeIndex`.
- Drops non‑numeric columns; errors if none remain.
- Coerces column names to integers representing maturities in months (e.g., "1 Mo", "12M", "60" -> 1, 12, 60). Errors if coercion fails.
- Enforces monthly frequency with no duplicates or gaps (allows 28–32 days between rows).
- Drops rows with any NaNs.
- Converts values to `float` and applies scaling.
- Issues a warning if any scaled yield is outside [-0.10, 0.50] (−10% to 50%).

About scaling:
- All yield values are multiplied by the `scale` argument immediately after conversion to `float` and before the sanity range check.
- Use `scale=0.01` if your Excel values are in percent (e.g., 5 becomes 0.05).

Example:

```python
# If running from this folder, Python can import rsim directly.
import rsim

yield_data = rsim.load_yield_data(
    filepath="FRB_H15.xlsx",
    min_date="2000-01-01",
    scale=0.01  # Excel file is in percent; convert to decimals
)
```

If you run from a different working directory, ensure this folder is on `sys.path`:

```python
import sys, os
project_root = os.path.dirname(__file__)  # path to this README's folder if inside a script
# Or use an explicit path, e.g.: project_root = "/path/to/quant-research-notebooks/yield-curve-simulation"
if project_root not in sys.path:
    sys.path.append(project_root)
import rsim
```

## Jupyter Notebook: `rate simulation.ipynb`

Example of loading data in the notebook:

```python
import rsim
yield_data = rsim.load_yield_data("FRB_H15.xlsx", min_date="2000-01-01", scale=0.01)
```

The notebook then:
- Fits a 3‑factor PCA model (level, slope, curvature) and reports explained variance.
- Trains a VAR(1) model on the factors and simulates future paths (e.g., 1000 paths × 120 months).
- Reconstructs simulated yield curves and visualizes various properties of simulated curves against historical data.

Notes and observations:
- Simulated short rates may dip negative more frequently due to symmetric VAR shocks.
- VAR(1) produces smoother dynamics and can under‑represent jump behavior seen in short rates.

Depending on your use case, consider more sophisticated models if these behaviors are undesirable.

## License

This project is released under the MIT License.
