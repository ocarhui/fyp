import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arima_gridsearch import ARIMAGridSearch

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

def prepare_and_analyze_data(borough_data):
    results_str = ""
    arima_params = {} 

    for borough in borough_data:
        current_borough = borough_data[borough]
        flats_data = current_borough[current_borough['Type'] == 'F'].copy()
        flats_data['Date'] = pd.to_datetime(flats_data['Date'])
        monthly_avg_price = flats_data.resample('M', on='Date')['Price'].mean().reset_index()
        print(monthly_avg_price.head())

        # Drop rows with NaN values in 'Price' column

        monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
        monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff']) 

        # Check if monthly_avg_price_cleaned has more than one unique value
        if monthly_avg_price_cleaned['Price_Diff'].nunique() > 1:
        # Proceed with the ADF test on the cleaned data
            try:
                adf_test = adfuller(monthly_avg_price_cleaned['Price_Diff'])
                adf_output = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
                for key, value in adf_test[4].items():
                    adf_output[f'Critical Value ({key})'] = value

                results_str += f"Key: {borough}\n{adf_output}\n\n"

                arima_grid_search = ARIMAGridSearch(p_range=range(0, 5), d_range=range(0, 2), q_range=range(0, 5))
                best_order, best_aic, best_model = arima_grid_search.search(monthly_avg_price_cleaned['Price_Diff'])
                arima_params[borough] = best_order
                results_str += f"Best ARIMA{best_order} Model for {borough} - AIC: {best_aic}\n"
                results_str += f"ARIMA Model Summary for {borough}:\n{best_model.summary()}\n\n"
                results_str += f"Monthly Average Price Data for {borough}:\n{monthly_avg_price_cleaned}\n\n"
            except Exception as e:
                results_str += f"Error performing ADF test on {borough}.\n\n"
    
        else:
            results_str += f"Cannot perform ADF test on constant series for {borough}.\n\n"

    return results_str, arima_params, monthly_avg_price

def save_results(results_str, file_path):
    with open(file_path, "w") as file:
        file.write(results_str)
    print(f"Results have been saved to {file_path}.")

def export_arima_params_to_csv(arima_params, file_path):
    # Convert the dictionary to a DataFrame for easier CSV export
    df_arima_params = pd.DataFrame(list(arima_params.items()), columns=['Borough', 'ARIMA_Params'])
    # Export the DataFrame to a CSV file
    df_arima_params.to_csv(file_path, index=False)
    print(f"ARIMA parameters have been exported to {file_path}.")

borough_data = load_data(DATA_DIR)
for borough in borough_data:
    print(borough)
    print(borough_data[borough].head())
#results_str, arima_params, monthly_avg_price = prepare_and_analyze_data(borough_data)
#save_results(results_str, "adf_test_results.txt")
#export_arima_params_to_csv(arima_params, "arima_params.csv")