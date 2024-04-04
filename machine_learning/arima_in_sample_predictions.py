import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "./data"


def prepare_features(flats_data):
    
    # Standardize numerical features
    numerical_features = flats_data[['Date','Price','EnergyEfficiency', 'HouseSize', 'NumOfRms', 'PricePer', 'BuildDate']]
    numerical_features_df = pd.DataFrame(numerical_features, columns=numerical_features.columns)

    # One-hot encode 'BuildDate'
    build_date = flats_data['BuildDate']

    # Combine all features
    features = pd.concat([build_date, numerical_features_df], axis=1)
    return features

def df_with_arima(borough, borough_data, arima_params):
    print(borough_data.columns)
    flats_data = borough_data.copy()
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
    arima_results_df = pd.DataFrame({'Month_Index': range(1, len(monthly_avg_price) + 1), 'ARIMA_Predictions': arima_in_sample_predictions})
    
    start_date = "1995-01"

    arima_results_df['Date'] = pd.to_datetime(start_date) + pd.to_timedelta((arima_results_df['Month_Index'] - 1) * 31, unit='D')
    arima_results_df['YearMonth'] = arima_results_df['Date'].dt.to_period('M')

    flats_data['YearMonth'] = flats_data['Date'].dt.to_period('M')

    combined_df = pd.merge(flats_data, arima_results_df[['ARIMA_Predictions', 'YearMonth']], on='YearMonth', how='left')
    combined_df = combined_df.drop(["YearMonth"], axis=1)

    return combined_df
    

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

if __name__ == '__main__':
    borough_data = load_data(DATA_DIR)

    # Read ARIMA parameters
    arima_file_path = 'arima_params.csv'
    df = pd.read_csv(arima_file_path)
    arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']

    for borough in borough_data:
        borough_df = borough_data[borough].copy()
        borough_df = df_with_arima(borough, borough_df, arima_params)
        final_file_path = f'./data/{borough}_combined.csv'
        borough_df.to_csv(final_file_path, index=False)

    """plt.figure(figsize=(20, 20), dpi=150)
    sns.heatmap(borough_df.drop(['ID', 'Postcode', 'Date', 'Street Number', 'Flat Number', 'Street Name', 'Area', 'Town', 'City', 'County'], axis=1).dropna().corr(), annot=False, cmap='coolwarm', center=0)
    plt.show()"""




