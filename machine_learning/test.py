import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preparation import preprocess_data, preprocess_data_without_arima, preprocess_data_with_date
import numpy as np
import pandas as pd

DATA_DIR = "./data"
ARIMA_DIR = "./arima_only_predictions"

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

def load_arima_only_predictions(data_dir):
    arima_only_predictions = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            arima_only_predictions[borough] = df
    return arima_only_predictions

def train_model(X_train, y_train, estimate=100, learn_rate=0.1, depth=5):
    """
    Trains an XGBoost regressor on the preprocessed data.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=estimate, learning_rate=learn_rate, max_depth = depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X):
    """
    Uses the trained model to predict housing prices.
    """
    predictions = model.predict(X)
    return predictions

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance using mean absolute error and root mean squared error.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return mae, rmse

def predict_monthly_averages(model, scaler, base_features):
    """
    Predicts the average house price for each month in 2022.

    Parameters:
    - model: The trained XGBoost model.
    - scaler: The StandardScaler object used during training.
    - base_features: A DataFrame containing a template of features for 2022.

    Returns:
    - A dictionary with months as keys and predicted prices as values.
    """
    predictions = {}
    for month in range(1, 13):
        base_features['Month'] = month
        # Scale features as per the training phase
        features_scaled = scaler.transform(base_features)
        predicted_price = model.predict(features_scaled)
        predictions[month] = predicted_price.mean()
    return predictions


def predict_for_parameters(model, scaler, feature_template, user_inputs):
    """
    Predicts house prices based on user-defined parameters.

    Parameters:
    - model: The trained model.
    - scaler: The scaler used in preprocessing.
    - feature_template: A template of features for prediction.
    - user_inputs: A dictionary containing user-specified values for certain features.

    Returns:
    - The predicted price for the given parameters.
    """
    # Update the feature_template dictionary with user_inputs
    for key, value in user_inputs.items():
        if key in feature_template:
            feature_template[key] = value
            
    # Convert the updated feature_template to a DataFrame for scaling
    features_df = pd.DataFrame([feature_template])
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    # Predict using the scaled features
    predicted_price = model.predict(features_scaled)

    return predicted_price[0]

def perform_grid_search(model_name, X, y):
    """
    Perform grid search to find the best hyperparameters for the XGBoost model.

    Parameters:
    - model_name: Name of the model for display purposes.
    - X: Feature matrix.
    - y: Target vector.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Use KFold for regression tasks
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    params = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        'max_depth': [3, 5, 7]
    }
    grid = GridSearchCV(xgb_model, params, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(f"{model_name} Grid Search Results:")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")  # Note: This score will be negative due to 'neg_mean_squared_error'

    return grid.best_params_

def plot_predictions(historical_data, predictions, borough_name):
    """
    Plots historical prices along with the predictions and saves the plot.

    :param historical_data: DataFrame with columns 'Date' and 'Price' for historical data
    :param predictions: DataFrame with columns 'Date' and 'Predicted_Price' for predictions
    :param borough_name: The name of the borough for title and filename
    """
    plt.figure(figsize=(12, 6))

    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    monthly_avg_price = historical_data.resample('M', on='Date')['Price'].mean().reset_index()
    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff']) 

    monthly_means = monthly_avg_price_cleaned.groupby(monthly_avg_price_cleaned['Date'].dt.to_period('M')).mean()
    monthly_means.index = monthly_means.index.to_timestamp()
    
    # Plot historical data
    if monthly_avg_price is not None and not monthly_avg_price.empty:
        plt.plot(monthly_means.index, monthly_means['Price'], label='Historical Prices', color='blue')
    
    # Plot predictions
    plt.plot(predictions['Date'], predictions['Predicted_Price'], color='red', label='Predicted Prices')
    plt.title(f'Price Predictions for {borough_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'./xgboost_predictions/{borough_name}_price_predictions.png')

borough_data = load_data(DATA_DIR)
integrated_data = pd.read_csv(DATA_DIR+'/other_data_2/integrated.csv')
arima_data = load_arima_only_predictions(ARIMA_DIR)
xgboost_params = pd.read_csv('xgboost_params.csv')

xgboost_params_dict = xgboost_params.set_index('Borough').to_dict(orient='index')
borough = "RM5"
if borough in borough_data:
    borough_df = borough_data[borough].copy()
    X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(borough, borough_df)

    # Assuming 'preprocess_data' returns scaled features and the target variable
    best_params = perform_grid_search(borough, X_train, y_train)
    xgboost_params_dict[borough] = best_params


