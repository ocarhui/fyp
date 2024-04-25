from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

arima_results = pd.read_csv('ensemble_combined_results.csv')
arima_results['Date'] = pd.to_datetime(arima_results['Date'])

# We will provide a statistical summary of the predictions, confidence intervals, and trends.
# The summary includes average predicted prices, the range of confidence intervals, and any noteworthy observations.

# Calculate summary statistics for predicted prices and confidence intervals across Greater London
top_50_boroughs = arima_results['borough'].unique()[:50]
# Filter the DataFrame for only the top 100 boroughs
subset_results = arima_results[arima_results['borough'].isin(top_50_boroughs)]
subset_results['MonthYear'] = arima_results['Date'].dt.strftime('%m-%Y')
subset_results = subset_results[subset_results['borough'] != 'N6']
subset_results = subset_results[subset_results['borough'] != 'W11']

# Pivot the subset for heatmap visualization
pivot_subset = subset_results.pivot(index='MonthYear', columns='borough', values='Mean_Predictions')

plt.figure(figsize=(50, 30)) 
# Generate the heatmap
heatmap = sns.heatmap(pivot_subset, cmap='coolwarm', annot=False)
plt.yticks(rotation=0, fontsize=8) 

# Rotate x-axis labels and set font sizes for better readability
plt.xticks(rotation=90, fontsize=8)
plt.title('Heatmap of Housing Price Predictions for the first 50 Postal Districts', fontsize=14)
plt.xlabel('Postal District', fontsize=8)
plt.ylabel('Date', fontsize=8)

# Save and show the heatmap
#plt.savefig('/mnt/data/heatmap_top_100_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
