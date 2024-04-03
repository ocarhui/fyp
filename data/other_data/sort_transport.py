import pandas as pd

# Load the datasets
transport_index_path = './data/other_data/Transport_Index.csv'  # Update this path
filtered_postcodes_path = './data/other_data/London_postcodes.csv'  # Update this path

transport_index_df = pd.read_csv(transport_index_path)
filtered_postcodes_df = pd.read_csv(filtered_postcodes_path)
filtered_postcodes_df = filtered_postcodes_df[['Postcode', 'Ward Code', 'London zone', 'Index of Multiple Deprivation', 'Nearest station', 'Distance to station', 'Average Income', 'IMD decile']]

merged_df = pd.merge(filtered_postcodes_df, transport_index_df, left_on='Ward Code', right_on='Ward Code', how='inner')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('./data/other_data/new_transport_index.csv', index=False)