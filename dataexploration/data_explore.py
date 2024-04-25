import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

data_dir = "./data/"
borough_data = load_data(data_dir)


df = borough_data['SW8'].copy()
ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
df['PTAL2021'] = df['PTAL2021'].map(ptal_mapping)
df['Date'] = pd.to_datetime(df['Date'])
df['Timestamp'] = df['Date'].astype('int64') // 10**9
postcode_mapping = {postcode: i for i, postcode in enumerate(df['Postcode'].unique())}
df['Postcode'] = df['Postcode'].map(postcode_mapping)
df['Age'] = df['Year'] - df['BuildDate']
df=df.rename(columns = {'Index of Multiple Deprivation':'IMD'})
df.drop(['ID','Date', 'Street Number', 'Flat Number', 'Street Name', 'Area', 'Town', 'City', 'County', 'Postcode Prefix', 'Type', 'Newbuild','Ownership', 'Nearest station', 'Year', 'BuildDate'], axis=1, inplace=True)
print(df.dtypes)

def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Variable Correlations of SW8 Postcode District')
    plt.xticks(rotation=90)
    plt.tight_layout() 
    plt.show()

plot_heatmap(df)




"""# Plot the graph as a bar chart
plt.bar(result['Prefix'], result['Value'], color='blue')
plt.xlabel('Postcode Districts')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions per Postcode District')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.ticklabel_format(style='plain', axis='y', useOffset=False) 
plt.show()
"""