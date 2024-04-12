import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preparation import preprocess_data, preprocess_data_without_arima, preprocess_data_with_date
import optuna

DATA_DIR = "./data"
ARIMA_DIR = "./arima_only_predictions"
XGBOOST_DIR = "./xgboost_predictions"
TF_DIR = "./tf_predictions"

def load_data(data_dir, arima):
    borough_data = {}
    if arima == False:
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                borough = filename.split('_')[0]
                df = pd.read_csv(file_path)
                borough_data[borough] = df
    else:
        for filename in os.listdir(data_dir):
            if filename.endswith('forecast.csv'):
                file_path = os.path.join(data_dir, filename)
                borough = filename.split('_')[0]
                df = pd.read_csv(file_path)
                borough_data[borough] = df
    return borough_data

def plot_predictions(historical_data, combined_df, borough_name):
    """
    Plots historical prices along with the predictions and saves the plot.

    :param historical_data: DataFrame with columns 'Date' and 'Price' for historical data
    :param predictions: DataFrame with columns 'Date' and 'Predicted_Price' for predictions
    :param borough_name: The name of the borough for title and filename
    """
    plt.figure(figsize=(12, 7))

    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    monthly_avg_price = historical_data.resample('M', on='Date')['Price'].mean().reset_index()
    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff']) 

    monthly_means = monthly_avg_price_cleaned.groupby(monthly_avg_price_cleaned['Date'].dt.to_period('M')).mean()
    monthly_means.index = monthly_means.index.to_timestamp()
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Plot historical data
    if monthly_avg_price is not None and not monthly_avg_price.empty:
        plt.plot(monthly_means.index, monthly_means['Price'], label='Historical Prices', color='blue')
    
    # Plot predictions
    plt.plot(combined_df['Date'], combined_df['ARIMA_Predictions'], color='red', label='ARIMA Predicted Prices', linewidth=2, markevery=5, alpha=0.7)
    plt.plot(combined_df['Date'], combined_df['XGBoost_Predictions'], color='green', label='XGBoost Predicted Prices', linewidth=2, markevery=5, alpha=0.7)
    plt.plot(combined_df['Date'], combined_df['TF_Predictions'], color='black', label='TF Predicted Prices', linewidth=2, markevery=5, alpha=0.7)
    plt.plot(combined_df['Date'], combined_df['Mean_Predictions'], color='purple', label='Ensemble Predicted Prices', linewidth=2, markevery=5, alpha=0.7)

    #plt.yscale('log')  # Apply logarithmic scale
    plt.title(f'Price Predictions for {borough_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./ensemble_prediction/{borough_name}_price_predictions.png')

def extract_tf_metrics(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    
    # Assuming the first line is in the format: [value1, value2, value3]
    try:
        values = eval(first_line)
    except:
        values = [1, 1, 1]
    
    
    # Mapping the extracted values to the respective metrics
    results_dict = {
        "tf_loss": values[0],
        "tf_mae": values[1],
        "tf_rmse": values[2],
    }
    
    return results_dict

def extract_arima_params(file_path):
    df = pd.read_csv(file_path)
    rmse = df['RMSE'][0]
    mae = df['MAE'][0]

    return rmse, mae
    
arima_bool = False
arima_true = True
data_df = load_data(DATA_DIR, arima_bool)
arima_df = load_data(ARIMA_DIR, arima_true)
xgboost_df = load_data(XGBOOST_DIR, arima_bool)
tf_df = load_data(TF_DIR, arima_bool)

for borough in data_df:
    print(borough)
    borough_df = data_df[borough]
    curr_arima_df = arima_df[borough]
    curr_xgboost_df = xgboost_df[borough]
    curr_tf_df = tf_df[borough]

    arima_predictions = []
    xgboost_predictions = []
    tf_predictions = []

    arima_rmse, arima_mae = extract_arima_params(f'{ARIMA_DIR}/{borough}_metrics.csv')

    xgboost_mae = curr_xgboost_df['mae'][0]
    xgboost_rmse = curr_xgboost_df['rmse'][0]

    tf_stats = extract_tf_metrics(f'{TF_DIR}/{borough}_analyse_results.txt')
    tf_mae = tf_stats['tf_mae']
    tf_rmse = tf_stats['tf_rmse']

    inv_mae_arima = 1 / arima_mae
    inv_rmse_arima = 1 / arima_rmse
    inv_mae_xgb = 1 / xgboost_mae
    inv_rmse_xgb = 1 / xgboost_rmse
    inv_mae_tf = 1 / tf_mae
    inv_rmse_tf = 1 / tf_rmse

    performance_arima = (inv_mae_arima + inv_rmse_arima) / 2
    performance_xgb = (inv_mae_xgb + inv_rmse_xgb) / 2
    performance_tf = (inv_mae_tf + inv_rmse_tf) / 2

    total_performance = performance_xgb + performance_tf + performance_arima
    weight_arima = (performance_arima / total_performance)
    weight_xgb = (performance_xgb / total_performance)
    weight_tf = (performance_tf / total_performance)

    assert len(arima_predictions) == len(xgboost_predictions) == len(tf_predictions)

    for i in range(len(curr_arima_df)):
        arima_predictions.append(curr_arima_df['Predicted_Price'][i])
        xgboost_predictions.append(curr_xgboost_df['Predicted_Price'][i])
        tf_predictions.append(curr_tf_df['Average_Price'][i])

    # Calculate the mean for each position
    means = [(arima_predictions[i] * weight_arima + xgboost_predictions[i] * weight_xgb + tf_predictions[i] * weight_tf) for i in range(len(arima_predictions))]

    combined_df = pd.DataFrame()
    combined_df['Date'] = curr_arima_df['Date']
    combined_df['ARIMA_Predictions'] = curr_arima_df['Predicted_Price']
    combined_df['XGBoost_Predictions'] = curr_xgboost_df['Predicted_Price']
    combined_df['TF_Predictions'] = curr_tf_df['Average_Price']
    combined_df['Mean_Predictions'] = means
    combined_df['weight_xgb'] = weight_xgb
    combined_df['weight_tf'] = weight_tf

    combined_df.to_csv(f'./ensemble_prediction/{borough}_predictions.csv', index=False)
    plot_predictions(borough_df, combined_df, borough)
    


