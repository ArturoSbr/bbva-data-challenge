# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Generate a fake target
def fake_target(df, target_name='y', seed=42):
    np.random.seed(seed)
    df = df.copy()

    # Lagged values (with fallback fill)
    df["lag_inpc_1"] = df["inpc"].shift(1).bfill()
    df["lag_cetes_3"] = df["cetes_1m"].shift(3).bfill()
    df["lag_fx_6"] = df["exchange_rate_usd"].shift(6).bfill()

    # Interactions and ratios
    df["fx_to_forecast"] = df["exchange_rate_usd"] / (df["official_interest_rate_usa"] + 1e-6)

    # Seasonality via calendar month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Trend (already defined as df['t'])
    df["t2"] = df["t"] ** 2

    # Generate target with noise and complexity
    df[target_name] = (
        0.6 * df["lag_inpc_1"] +
        0.4 * np.log1p(np.abs(df["lag_fx_6"])) +
        0.3 * df["fx_to_forecast"] +
        1.0 * df["month_sin"] +
        0.01 * df["t2"] +
        np.random.normal(0, 0.5, size=len(df))
    )

    return df


# Dual y-axis plot
def plot_dual_y(
    df,
    series1: str,
    series2: str,
    x: str,
    xticks_every: int = 12,
    color1: str = 'C0',
    color2: str = 'C1',
    grid: bool = True,
    filename: str = None
):
    """Plot two time series on dual y-axes.

    Parameters:
    - df: pandas DataFrame
    - series1: Column name for the left y-axis
    - series2: Column name for the right y-axis
    - x: Column name for the x-axis
    - xticks_every: Frequency of x-ticks (in periods)
    - color1: Line color for series1
    - color2: Line color for series2
    - grid: Whether to display grid
    - filename: Path to save
    """
    fig, ax1 = plt.subplots()
    
    label1 = series1.replace('_', ' ').title()
    label2 = series2.replace('_', ' ').title()

    # Plot series1 on left y-axis
    ax1.plot(df[series1], color=color1)
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params(axis='y', color=color1)
    ax1.grid(visible=grid)

    # Set custom x-ticks
    xticks_idx = [i for i in range(len(df)) if i % xticks_every == 0]
    ax1.set_xticks(xticks_idx)
    ax1.set_xticklabels(df[x].iloc[xticks_idx], rotation=45)

    # Plot series2 on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(df[series2], color=color2)
    ax2.set_ylabel(label2, color=color2)
    ax2.tick_params(axis='y', color=color2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
