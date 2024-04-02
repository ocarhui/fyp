import pandas as pd

# Load the datasets
imd_path = './data/other_data/IMD_WARD.csv'  # Update this path
filtered_postcodes_path = './data/other_data/filtered_postcodes.csv'  # Update this path

imd_df = pd.read_csv(imd_path)
filtered_postcodes_df = pd.read_csv(filtered_postcodes_path)

# Create a mapping from wd11cd to pcds
ward_to_postcode_map = filtered_postcodes_df.set_index('wd11cd')['pcds'].to_dict()

# Replace the Ward Code in Transport Index with Postcode
# Assuming the column in Transport Index DataFrame containing the Ward Code is named "WardCode"
# Update "WardCode" with the actual column name if it's different
imd_df['Postcode'] = imd_df['Ward Code'].map(ward_to_postcode_map)
imd_df = imd_df[['Postcode', 'IMD Extent %', 'IMD average score', 'Income score', 'Employment score']]

# Save the updated DataFrame to a new CSV file
imd_df.to_csv('./data/other_data/updated_imd.csv', index=False)