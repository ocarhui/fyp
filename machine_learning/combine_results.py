from math import sqrt
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

DATA_DIR = "./arima_only_predictions"

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('metrics.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

borough_data = load_data(DATA_DIR)
combined_df = pd.DataFrame()
for borough in borough_data:
    borough_data[borough]['borough'] = borough
    combined_df = pd.concat([combined_df, borough_data[borough]], ignore_index=True)

combined_df.to_csv('arima_metrics.csv', index=False)