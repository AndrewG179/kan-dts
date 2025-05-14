import numpy as np
import pandas as pd

def generate_trend(n, slope=0.1):
    return np.arange(n) * slope

def generate_seasonality(n, period=50, amplitude=1):
    return amplitude * np.sin(np.arange(n) * 2 * np.pi / period)

def generate_noise(n, scale=0.3):
    return np.random.normal(0, scale, size=n)

def create_time_series(n=500, components=["trend", "seasonality"], noise=True):
    series = np.zeros(n)
    if "trend" in components:
        series += generate_trend(n)
    if "seasonality" in components:
        series += generate_seasonality(n)
    if noise:
        series += generate_noise(n)
    return pd.DataFrame({"y": series})