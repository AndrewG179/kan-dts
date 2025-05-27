import pandas as pd
import numpy as np
from statsmodels.datasets import sunspots, co2

def load_small_dataset(name):
    if name == "sunspots":
        return sunspots.load_pandas().data['SUNACTIVITY'].values.reshape(-1, 1)

    elif name == "daily_temp":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
        df = pd.read_csv(url)
        return df['Temp'].values.reshape(-1, 1)

    elif name == "co2":
        return co2.load_pandas().data['co2'].dropna().values.reshape(-1, 1)

    else:
        raise ValueError(f"Unknown dataset: {name}")