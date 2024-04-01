import pandas as pd
import os

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

def process_dataframe(df):
    # Adjust 'builddate' based on the instructions
    df['BuildDate'] = df['BuildDate'].replace({'England and Wales: 2007 onwards': 2007, 'England and Wales: 2012 onwards': 2012})
    # Ensure builddate and buydate are numeric for comparison
    df['BuildDate'] = pd.to_numeric(df['BuildDate'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # Apply the condition to make 'builddate' as 0 if it's larger than 'buydate'
    df.loc[df['BuildDate'] > df['Year'], 'BuildDate'] = 0
    # Handle 'nodata' similar to the processing for 'builddate'
    df.loc[df['BuildDate'].isna(), 'BuildDate'] = 0

borough_data = load_data(DATA_DIR)

for borough in borough_data:
    process_dataframe(borough_data[borough])
    borough_data[borough].to_csv(f"{DATA_DIR}/{borough}_combined.csv", index=False)