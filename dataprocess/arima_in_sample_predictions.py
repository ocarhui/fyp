import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

DATA_DIR = "./data"


def prepare_features(flats_data):
    # One-hot encode 'BuildDate'
    build_date = flats_data['BuildDate']

    # Standardize numerical features
    numerical_features = flats_data[['Price','EnergyEfficiency', 'HouseSize', 'NumOfRms', 'PricePer', 'BuildDate']]
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    numerical_features_df = pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns)

    # Combine all features
    features = pd.concat([build_date, numerical_features_df], axis=1)
    return features

def df_with_arima(borough, borough_data, arima_params):
    flats_data = borough_data[borough_data['Type'] == 'F'].copy()
    flats_data['Date'] = pd.to_datetime(flats_data['Date'])
    monthly_avg_price = flats_data.resample('ME', on='Date')['Price'].mean().reset_index()
    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff'])

    features = prepare_features(flats_data)
    
    # Fit ARIMA model
    p, d, q = int(arima_params[borough][1]), int(arima_params[borough][4]), int(arima_params[borough][7])
    arima_model = ARIMA(monthly_avg_price_cleaned['Price_Diff'], order=(p, d, q))
    arima_model_fit = arima_model.fit()
    arima_in_sample_predictions = arima_model_fit.predict(start=0, end=len(monthly_avg_price) - 1)

    prediction_dates = pd.date_range(start="1995-01", periods=324, freq='ME')
    arima_predictions_series = pd.Series(arima_in_sample_predictions, index=prediction_dates)

    combined_df = pd.concat([features, arima_predictions_series], axis=1)

    combined_df.to_csv('out.csv')
    

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

# Read ARIMA parameters
arima_file_path = 'arima_params.csv'
df = pd.read_csv(arima_file_path)
arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']

df_with_arima('SW11', borough_data["SW11"], arima_params)




