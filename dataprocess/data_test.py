import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

DATA_DIR = "./data"

borough_data = {}

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(DATA_DIR, filename)
        borough = filename.split('_')[0]
        
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        borough_data[borough] = df

results_str = "" 

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
        except Exception as e:
            results_str += f"Error performing ADF test on {borough}.\n\n"
    else:
        results_str += f"Cannot perform ADF test on constant series for {borough}.\n\n"

    
with open("adf_test_results.txt", "w") as file:
    file.write(results_str)

print("ADF test results have been saved to adf_test_results.txt.")
    

        
        
       