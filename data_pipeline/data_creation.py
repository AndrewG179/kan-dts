import numpy as np
import pandas as pd

# ---- individual component helpers ----
def generate_trend(n, slope=0.1):
    return np.arange(n) * slope

def generate_seasonality(n, period=10, amplitude=1):
    return amplitude * np.sin(np.arange(n) * 2*np.pi / period)

def generate_noise(n, scale=0.1):
    return np.random.normal(0, scale, size=n)

# ---- high-level factory ----
def create_time_series(n=500, components=("trend","seasonality"),
                       noise=True, scale=0.1):
    series = np.zeros(n)
    if "trend"      in components: series += generate_trend(n)
    if "seasonality" in components: series += generate_seasonality(n)
    if noise: series += generate_noise(n, scale)
    return pd.DataFrame({"y": series})

# ---- convenience wrapper used by batch runner ----
def load_synthetic(kind="trend_season", noise_std=0.1, n=500):
    if kind == "trend_only":
        return create_time_series(n, components=("trend",), noise=True,
                                  scale=noise_std)
    if kind == "season_only":
        return create_time_series(n, components=("seasonality",), noise=True,
                                  scale=noise_std)
    # default: both
    return create_time_series(n, components=("trend","seasonality"),
                              noise=True, scale=noise_std)