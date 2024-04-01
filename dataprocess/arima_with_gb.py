import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

def prepare_features(flats_data):

    # Process 'Newbuild' into a binary variable
    isNewBuild = flats_data['Newbuild']

    # One-hot encode 'BuildDate'
    build_date = flats_data['BuildDate']

    # Standardize numerical features
    numerical_features = flats_data[['EnergyEfficiency', 'HouseSize', 'NumOfRms', 'PricePer']]
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    numerical_features_df = pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns)

    # Combine all features
    features = pd.concat([build_date, numerical_features_df, isNewBuild], axis=1)
    return features

# Load data
borough_data = load_data(DATA_DIR)

# Read ARIMA parameters
arima_file_path = 'arima_params.csv'
df = pd.read_csv(arima_file_path)
arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']

# Process each borough
for borough in borough_data:
    current_borough = borough_data[borough]
    flats_data = current_borough[current_borough['Type'] == 'F'].copy()
    flats_data['Date'] = pd.to_datetime(flats_data['Date'])
    monthly_avg_price = flats_data.resample('M', on='Date')['Price'].mean().reset_index()

    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff'])
    
    # Fit ARIMA model
    p, d, q = int(arima_params[borough][1]), int(arima_params[borough][4]), int(arima_params[borough][7])
    arima_model = ARIMA(monthly_avg_price_cleaned['Price_Diff'], order=(p, d, q))
    arima_model_fit = arima_model.fit()

    # Generate ARIMA in-sample predictions
    arima_in_sample_predictions = arima_model_fit.predict(start=0, end=len(monthly_avg_price) - 1)
    
    # Prepare additional features
    additional_features = prepare_features(flats_data)

    # Combine ARIMA predictions with additional features
    features = additional_features.copy()
    print(features.size)
    print(arima_in_sample_predictions)
    features['ARIMA_Prediction'] = arima_in_sample_predictions.values
    
    # Define target
    target = monthly_avg_price['Price']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    # Train Gradient Boosting model
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)

    # Predict residuals for the test set
    gbr_predicted_residuals = gbr.predict(X_test)

    # Combine ARIMA and Gradient Boosting predictions for the final forecast
    final_forecasts = arima_model_fit.forecast(steps=len(y_test)) + gbr_predicted_residuals
    print(f"Final forecasts for {borough}:\n{final_forecasts}")