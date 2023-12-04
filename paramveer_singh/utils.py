import numpy as np
import pandas as pd

# define function to standardize numerical features of dataframe
def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    if use_log:
        series = np.log(series)
    return (series - np.mean(series))/np.std(series)