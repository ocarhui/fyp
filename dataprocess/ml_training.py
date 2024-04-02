import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from arima_in_sample_predictions import df_with_arima
import numpy as np

DATA_DIR = "./data"

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

borough_data = load_data(DATA_DIR)
arima_file_path = 'arima_params.csv'
df = pd.read_csv(arima_file_path)
arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']


for borough in borough_data:
    borough_df = df_with_arima(borough, borough_data, arima_params)
    