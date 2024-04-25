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

def eval_arima(borough, monthly_avg_price_cleaned):
    training_arima = pd.DataFrame(monthly_avg_price_cleaned.iloc[:int(len(monthly_avg_price_cleaned) * 0.8)])
    testing_arima = pd.DataFrame(monthly_avg_price_cleaned.iloc[int(len(monthly_avg_price_cleaned) * 0.8):])


    arima_model = ARIMA(training_arima['Price_Diff'], order=(p_range, d_range, q_range))
    arima_model_fit = arima_model.fit()

    forecast_result = arima_model_fit.get_forecast(steps=len(testing_arima))

    forecast_mean = forecast_result.predicted_mean
    cumulative_forecast = np.cumsum(forecast_mean)

    rmse = np.sqrt(mean_squared_error(testing_arima['Price_Diff'], cumulative_forecast))
    mae = mean_absolute_error(testing_arima['Price_Diff'], cumulative_forecast)

    metrics_df = pd.DataFrame({'RMSE': [rmse], 'MAE': [mae]})
    metrics_df.to_csv(f'./arima_only_predictions/{borough}_metrics.csv', index=False)


borough_data = load_data(DATA_DIR)

arima_file_path = 'arima_params.csv'
df = pd.read_csv(arima_file_path)
arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']

last_real_value = {}
output = []

for borough in borough_data:
    current_borough = borough_data[borough].copy()
    flats_data = current_borough
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


        
        forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq='M')

        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Difference': cumulative_forecast,
            'Predicted_Price': absolute_forecast,
            'Lower_Confidence_Interval': forecast_conf_int.iloc[:, 0] + last_real_value,
            'Upper_Confidence_Interval': forecast_conf_int.iloc[:, 1] + last_real_value
        })

        
        

        # Print the forecast results
        print(f"Forecasted Monthly Average Price in {borough} {last_month_mean}:")
        for index, row in forecast_df.iterrows():
            print(f"{row['Date'].strftime('%Y-%m')}: £{row['Predicted_Price']:.2f} (Confidence Interval: £{row['Lower_Confidence_Interval']:.2f} - £{row['Upper_Confidence_Interval']:.2f})")

        # Export the forecast DataFrame to a CSV file
        output_filename = f"./arima_only_predictions/{borough}_forecast.csv"
        forecast_df.to_csv(output_filename, index=False)
        print(f"Forecast saved to {output_filename}")

        print(f"Forecasted Monthly Average Price of Flats in {borough} {last_month_mean}:")

        for date, price, price2 in zip(forecast_dates, absolute_forecast, cumulative_forecast):
            print(f"{date.strftime('%Y-%m')}: £{price:.2f} and £{price2:.2f}")


        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_means.index, monthly_means['Price'], label='Historical Monthly Mean Price', color='blue')
        plt.plot(forecast_dates, absolute_forecast, color='red', label='Forecasted Monthly Mean Price')  # This is your forecast mean
        plt.fill_between(forecast_dates, forecast_conf_int.iloc[:, 0] + last_real_value,
                        forecast_conf_int.iloc[:, 1] + last_real_value, color='pink', alpha=0.5, label='Confidence Interval')
        plt.title(f'Forecast of Monthly Average Price in {borough}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'./arima_only_predictions/{borough}_monthly_avg_price.png')