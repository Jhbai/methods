import pandas as pd
import numpy as np

def analyze_pot(data: pd.Series, threshold: float) -> tuple[pd.Series, np.ndarray]:
    """
    data: pd.Series(n_batch, )
    threshold: float
    return:
        exceedances: pd.Series
        excesses: np.ndarray
        excesses_idx: np.ndarray
    """
    # ----- Find all the points that exceeds the threshold ----- #
    exceedances = data[data > threshold]
    if exceedances.empty:
        return pd.Series(dtype=float), np.array([])
    excesses = exceedances.values - threshold
    exceedance_indices = exceedances.index.values
    return exceedances, excesses, exceedance_indices

def get_empirical_cdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    data: pd.Series(n_batch, ), that is exceedances.values - threshold 
    return:
        x: np.ndarray, sorted array
        y: np.ndarray, the corresponding ECDF probability values
    """
    if data.size == 0:
        return np.array([]), np.array([])
    x = np.sort(data)
    n = x.shape[0]
    y = np.arange(1, n + 1) / n
    return x, y
