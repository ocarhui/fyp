import pandas as pd
import os

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

def expand_postcodes_for_merge(df, postcode_col='Postcode'):
    df['Postcode_Long'] = df[postcode_col]
    df['Postcode_Medium'] = df['Postcode_Long'].apply(lambda x: x[:-1] if len(x) > 3 else x)
    df['Postcode_Short'] = df['Postcode_Long'].apply(lambda x: x[:-2] if len(x) > 3 else x)
# Load the CSV file, indicating there is no header
df = pd.read_csv('./data/original_data/pp-2022.csv', header=None)

# Generate new column names with a prefix
prefix = 'prefix_'  # Define the prefix you want to add
new_columns = [prefix + str(i) for i in range(df.shape[1])]  # Create a list of new column names with the prefix

# Assign these new column names to the DataFrame
new_columns[3] = 'Postcode'
new_columns[1] = 'Price'
new_columns[2] = 'Date'
df.columns = new_columns
df.dropna(inplace=True)
df['Postcode_Prefix'] = df['Postcode'].apply(lambda x: x[:-3] if len(x) > 4 else x)
print(df['Postcode_Prefix'].head())

DATA_DIR = "./data"
borough_data = load_data(DATA_DIR)

for borough in borough_data:
    rows_with_substring1 = []
    rows_with_substring2 = []
    dual = False
    if borough == "BR5BR8":
        dual = True
        rows_with_substring1 = df[df['Postcode_Prefix'].str.contains('BR5')]
        rows_with_substring2 = df[df['Postcode_Prefix'].str.contains('BR8')]
    if borough == "E15E20":
        dual = True
        rows_with_substring1 = df[df['Postcode_Prefix'].str.contains('E15')]
        rows_with_substring2 = df[df['Postcode_Prefix'].str.contains('E20')]
    if borough == "EN3EN8":
        dual = True
        rows_with_substring1 = df[df['Postcode_Prefix'].str.contains('EN3')]
        rows_with_substring2 = df[df['Postcode_Prefix'].str.contains('EN8')]
    if borough == "IG1IG4":
        dual = True
        rows_with_substring1 = df[df['Postcode_Prefix'].str.contains('IG1')]
        rows_with_substring2 = df[df['Postcode_Prefix'].str.contains('IG4')]
    else:
        rows_with_substring1 = df[df['Postcode_Prefix'].str.contains(borough)]

    if dual == True:
        rows_with_substring1 = pd.concat([rows_with_substring1, rows_with_substring2])
        dual = False
    print(borough)
    rows_with_substring1['Date'] = pd.to_datetime(rows_with_substring1['Date'])
    monthly_avg_price = rows_with_substring1.resample('M', on='Date')['Price'].mean().reset_index()

    monthly_avg_price.to_csv(f'./original_2022_avg_prices/{borough}_original.csv', index=False)

        
    
    
    



