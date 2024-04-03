import pandas as pd

# Load the datasets
imd_index_path = './data/other_data/IMD_WARD.csv'  # Update this path
filtered_postcodes_path = './data/other_data/include_transport_index.csv'  # Update this path

imd_index_df = pd.read_csv(imd_index_path)
filtered_postcodes_df = pd.read_csv(filtered_postcodes_path)

merged_df = pd.merge(filtered_postcodes_df, imd_index_df, left_on='wd11cd', right_on='Ward Code', how='inner')
merged_df = merged_df[['pcds', 'AvPTAI2015', 'PTAL', 'IMD Extent %', 'IMD average score', 'Income score', 'Employment score']]

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('./data/other_data/transport_imd.csv', index=False)