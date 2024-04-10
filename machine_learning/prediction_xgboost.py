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

for borough in borough_data:
    borough_df = borough_data[borough].copy()
    print(f"processing {borough}")
    X_train, X_test, y_train, y_test, scaler, postcode_mapping = preprocess_data_with_date(borough, borough_df)

    learning_rate = xgboost_params[xgboost_params['Borough'] == borough]['learning_rate'].values[0]
    max_depth = xgboost_params[xgboost_params['Borough'] == borough]['max_depth'].values[0]
    n_estimators = xgboost_params[xgboost_params['Borough'] == borough]['n_estimators'].values[0]

    model = train_model(X_train, y_train, int(n_estimators), learning_rate, int(max_depth))
    postcode_query = next(iter(postcode_mapping))
    ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}

    feature_template = {
    #'Postcode': postcode_mapping[postcode_query], 
    'Postcode': postcode_mapping[postcode_query], # placeholder value’
    'HouseSize': borough_df['HouseSize'].mean(),  # average size in square meters
    'EnergyEfficiency': borough_df['EnergyEfficiency'].mean(),  # average energy efficiency rating
    'BuildDate': borough_df['BuildDate'].mean(),
    'Distance to station': integrated_data[integrated_data['Postcode'] == postcode_query]['Distance to station'].values[0],
    'Average Income': integrated_data[integrated_data['Postcode'] == postcode_query]['Average Income'].values[0],
    'IMD decile': integrated_data[integrated_data['Postcode'] == postcode_query]['IMD decile'].values[0],
    'NumOfRms': borough_df['NumOfRms'].mean(),     # average number of rooms
    'ARIMA_Predictions': 94048.27005310703,
    'Timestamp': int(datetime.datetime(2022,1,1).timestamp()),
    'PTAL2021': ptal_mapping[integrated_data[integrated_data['Postcode'] == postcode_query]['PTAL2021'].values[0]],
    'London zone': integrated_data[integrated_data['Postcode'] == postcode_query]['London zone'].values[0]
    }
    
    """user_input = {
        'Postcode': postcode_mapping['SW8 2FZ'],  # placeholder value’
        'HouseSize': 70,  # average size in square meters
        'EnergyEfficiency': 84,  # average energy efficiency rating
        'BuildDate': 2016,
        'NumOfRms': 2,     # average number of rooms
    }
    print(predict_for_parameters(model, scaler, feature_template, user_input))"""

    monthly_prices = {month: [] for month in range(1, 13)}

    for postcode in postcode_mapping:
        print('calculating for postcode:', postcode)
        for month in range(1, 13):
            # Prepare the feature set for each postcode and month
            current_df = borough_df[borough_df['Postcode'] == postcode_mapping[postcode]]
            features = feature_template.copy()  # Your base features excluding 'Month' and 'Postcode'
            features['Postcode'] = postcode_mapping[postcode] # Assuming postcode mapping to integer
            features['Distance to station'] = current_df['Distance to station'].values[0]
            features['Average Income'] = current_df['Average Income'].values[0]
            features['IMD decile'] = current_df['IMD decile'].values[0]
            features['PTAL2021'] = current_df['PTAL2021'].values[0]
            features['London zone'] = current_df['London zone'].values[0]
            features['Timestamp'] = int(datetime.datetime(2022,month,1).timestamp()), 
            features['Timestamp'] = pd.to_numeric(features['Timestamp'], downcast='integer')
            features['ARIMA_Predictions'] = arima_data[borough]['Difference'][month-1]
            features_df = pd.DataFrame([features])  # Convert to DataFrame
            print(features_df['Timestamp'].dtype)
            features_scaled = scaler.transform(features_df)  # Scale features

            # Predict the price
            predicted_price = model.predict(features_scaled)[0]
            monthly_prices[month].append(predicted_price)

    mae, rmse = evaluate_model(model, X_test, y_test)

    last_date = borough_df['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq='M')
    # Average the predictions for each month to get the borough-wide average
    average_monthly_prices = {month: np.mean(prices) for month, prices in monthly_prices.items()}

    average_prices_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': average_monthly_prices.values(),
            'rmse': rmse,
            'mae': mae
        })
    #average_prices_df = pd.DataFrame(list(average_monthly_prices.items()), columns=['Month', 'Average Price'])

    # Export to CSV
    average_prices_df.to_csv(f'xgboost_predictions/{borough}_forecast_xgboost.csv', index=False)
    plot_predictions(borough_df, average_prices_df, borough)


"""xgboost_params_dict = xgboost_params.set_index('Borough').to_dict(orient='index')

for borough in borough_data:
    borough_df = borough_data[borough].copy()
    X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(borough, borough_df)

    # Assuming 'preprocess_data' returns scaled features and the target variable
    best_params = perform_grid_search(borough, X_train, y_train)
    xgboost_params_dict[borough] = best_params

# Export best_params to CSV
xgboost_param_df = pd.DataFrame.from_dict(xgboost_params_dict)
xgboost_param_df.transpose().reset_index().rename(columns={'index': 'Borough'}).to_csv('xgboost_params.csv', index=False)"""
