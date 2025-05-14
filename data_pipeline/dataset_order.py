from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(X, y, test_ratio=0.2):
    return train_test_split(X, y, test_size=test_ratio, random_state=42)