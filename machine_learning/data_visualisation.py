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
        
for key in borough_data:
    current_borough = borough_data[key]
    flats_data = current_borough[current_borough['Type'] == 'F'].copy()
    flats_data['Date'] = pd.to_datetime(flats_data['Date'])
    monthly_avg_price = flats_data.resample('M', on='Date')['Price'].mean().reset_index()
    print(monthly_avg_price.head()) 
    
    # time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg_price['Date'], monthly_avg_price['Price'], marker='o', linestyle='-', color='blue')
    plt.title('Monthly Average Price of Flats in ' + key)
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.savefig(f'./plots/{key}_monthly_avg_price.png')