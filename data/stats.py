import os
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

borough_dict = {}
total = 0

for borough in borough_data:
    borough_dict[borough] = len(borough_data[borough])

df = pd.DataFrame(list(borough_dict.items()), columns=['Postcode', 'Value'])
df['Prefix'] = df['Postcode'].apply(lambda x: re.match(r'[A-Za-z]+', x).group())
result = df.groupby('Prefix')['Value'].sum().reset_index()
sum = result['Value'].sum()
print(sum)


# Plot the graph as a bar chart
plt.bar(result['Prefix'], result['Value'], color='blue')
plt.xlabel('Postcode Districts')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions per Postcode District')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.ticklabel_format(style='plain', axis='y', useOffset=False) 
plt.show()
