import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from data_preparation import preprocess_data
from prediction_xgboost import load_arima_only_predictions
from statsmodels.tsa.arima.model import ARIMA
import prediction_xgboost
import prediction_tensorflow



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

def train_meta_model(base_predictions, y_train):
    # Train a meta-model on the predictions made by base models
    meta_model = LinearRegression().fit(base_predictions, y_train)
    return meta_model


borough_data = load_data(DATA_DIR)
integrated_data = pd.read_csv(DATA_DIR+'/other_data_2/integrated.csv')
arima_params = pd.read_csv("arima_params.csv").set_index('Borough').to_dict()['ARIMA_Params']
arima_data = load_arima_only_predictions(ARIMA_DIR)
xgboost_params = pd.read_csv('xgboost_params.csv')
tf_params = pd.read_csv("tf_params.csv")

def arima_model(borough, borough_data):
    current_borough = borough_data[borough]
    flats_data = current_borough[current_borough['Type'] == 'F'].copy()
    flats_data['Date'] = pd.to_datetime(flats_data['Date'])
    monthly_avg_price = flats_data.resample('M', on='Date')['Price'].mean().reset_index()

    # Drop rows with NaN values in 'Price' column

    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff']) 

    if monthly_avg_price_cleaned['Price_Diff'].nunique() > 1:
        current_borough = borough_data[borough]
        p_range = int(arima_params[borough][1])
        d_range = int(arima_params[borough][4])
        q_range = int(arima_params[borough][7])

        arima_model = ARIMA(monthly_avg_price_cleaned['Price_Diff'], order=(p_range, d_range, q_range))
        arima_model_fit = arima_model.fit()

        forecast_result = arima_model_fit.get_forecast(steps=12)

        forecast_mean = forecast_result.predicted_mean
        forecast_conf_int = forecast_result.conf_int()

        last_date = monthly_avg_price_cleaned['Date'].iloc[-1]
        last_month_mean = monthly_avg_price_cleaned[monthly_avg_price_cleaned['Date'].dt.month == last_date.month]['Price'].mean()
        last_real_value = monthly_avg_price['Price'].iloc[-1] 

        monthly_means = monthly_avg_price_cleaned.groupby(monthly_avg_price_cleaned['Date'].dt.to_period('M')).mean()

        # Convert back to datetime for plotting
        monthly_means.index = monthly_means.index.to_timestamp()


        cumulative_forecast = np.cumsum(forecast_mean)  # Cumulative sum of forecasted differences
        absolute_forecast = last_real_value + cumulative_forecast  # Add to last real value

        return absolute_forecast


for borough in borough_data:

    borough_df = borough_data[borough].copy()
    X_train, X_test, y_train, y_test, scaler, postcode_mapping = preprocess_data(borough, borough_df)
    print(f"processing {borough}")
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    arima_prediction = arima_model(borough, borough_data)
    arima_prediction = list(arima_prediction)

    x_learning_rate = xgboost_params[xgboost_params['Borough'] == borough]['learning_rate'].values[0]
    x_max_depth = xgboost_params[xgboost_params['Borough'] == borough]['max_depth'].values[0]
    x_n_estimators = xgboost_params[xgboost_params['Borough'] == borough]['n_estimators'].values[0]
    
    xgboost_model = prediction_xgboost.train_model(X_train, y_train, int(x_n_estimators), x_learning_rate, int(x_max_depth))
    
    tf_model, tf_history = prediction_tensorflow.train_model_with_params(X_train, y_train, tf_params, borough, epochs=100, validation_split=0.2)

    postcode_query = next(iter(postcode_mapping))

    ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
    
    xgboost_monthly_prices = {month: [] for month in range(1, 13)}

    for postcode in postcode_mapping:
        print('calculating for postcode:', postcode)
        for month in range(1, 13):
            features = {
                'Postcode': postcode_mapping[postcode],
                'HouseSize': borough_df['HouseSize'].mean(),
                'EnergyEfficiency': borough_df['EnergyEfficiency'].mean(),
                'BuildDate': borough_df['BuildDate'].mean(),
                'Distance to station': borough_df['Distance to station'].values[0],
                'Average Income': borough_df['Average Income'].values[0],
                'IMD decile': borough_df['IMD decile'].values[0],
                'NumOfRms': borough_df['NumOfRms'].mean(),
                'ARIMA_Predictions': arima_data[borough]['Difference'][month-1],
                'Month': month,
                'PTAL2021': borough_df['PTAL2021'].values[0],
                'London zone': borough_df['London zone'].values[0]
            }
            features_df = pd.DataFrame([features]) 
            features_scaled = scaler.transform(features_df)  

            # Predict the price
            predicted_price = xgboost_model.predict(features_scaled)[0]
            xgboost_monthly_prices[month].append(predicted_price)
    
    xg_boost_average_prices = {month: np.mean(prices) for month, prices in xgboost_monthly_prices.items()}
    xg_boost_final_avg_prices = list(xg_boost_average_prices.values())


    tf_avg_prices = []

    for month in range(1,13):
        print("calculating tf")
        features_list = []

        for postcode in postcode_mapping:
            current_df = borough_df[borough_df['Postcode'] == postcode_mapping[postcode]]

            features = {
                'Postcode': postcode_mapping[postcode],
                'HouseSize': borough_df['HouseSize'].mean(),
                'EnergyEfficiency': borough_df['EnergyEfficiency'].mean(),
                'BuildDate': borough_df['BuildDate'].mean(),
                'Distance to station': current_df['Distance to station'].values[0],
                'Average Income': current_df['Average Income'].values[0],
                'IMD decile': current_df['IMD decile'].values[0],
                'NumOfRms': borough_df['NumOfRms'].mean(),
                'ARIMA_Predictions': arima_data[borough]['Difference'][month-1],
                'Month': month,
                'PTAL2021': current_df['PTAL2021'].values[0],
                'London zone': current_df['London zone'].values[0]
            }

        features_list.append(features)
        features_df = pd.DataFrame(features_list)
        X_pred = features_df
        X_pred_scaled = scaler.transform(X_pred)
        predictions = prediction_tensorflow.make_predictions(tf_model, X_pred_scaled)
        avg_price = np.mean(predictions)
        tf_avg_prices.append(avg_price)
    
    print(arima_prediction)
    print(xg_boost_final_avg_prices)
    print(tf_avg_prices)
    base_predictions = np.column_stack((arima_prediction, xg_boost_final_avg_prices, tf_avg_prices))
    meta_model = train_meta_model(base_predictions, y_test)
    ensemble_prediction = meta_model.predict(base_predictions, y_train)
    results_df = pd.DataFrame({
        'Actual': y_train,  # Or y_test, depending on what you're comparing against
        'Predicted': ensemble_prediction
    })

    # Export the DataFrame to a CSV file
    csv_file_path = f'./ensemble_predicions/{borough}_predicion.csv'
    results_df.to_csv(csv_file_path, index=False)
    
    



    
            



