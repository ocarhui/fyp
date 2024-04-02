import pandas as pd

# Load the datasets
transport_index_path = './data/other_data/Transport_Index.csv'  # Update this path
filtered_postcodes_path = './data/other_data/filtered_postcodes.csv'  # Update this path

transport_index_df = pd.read_csv(transport_index_path)
filtered_postcodes_df = pd.read_csv(filtered_postcodes_path)

# Create a mapping from wd11cd to pcds
ward_to_postcode_map = filtered_postcodes_df.set_index('wd11cd')['pcds'].to_dict()

# Replace the Ward Code in Transport Index with Postcode
# Assuming the column in Transport Index DataFrame containing the Ward Code is named "WardCode"
# Update "WardCode" with the actual column name if it's different
transport_index_df['Postcode'] = transport_index_df['Ward Code'].map(ward_to_postcode_map)
transport_index_df = transport_index_df.drop(['Ward Code', 'Ward Name'], axis=1)

# Save the updated DataFrame to a new CSV file
transport_index_df.to_csv('./data/other_data/updated_Transport_Index.csv', index=False)