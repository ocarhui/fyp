import pandas as pd

file_paths = {
    '1995': './data/pp-cleaned1995.csv',
    '1996': './data/pp-cleaned1996.csv',
    '1997': './data/pp-cleaned1997.csv',
    '1998': './data/pp-cleaned1998.csv',
    '1999': './data/pp-cleaned1999.csv',
    '2000': './data/pp-cleaned2000.csv',
    '2001': './data/pp-cleaned2001.csv',
    '2002': './data/pp-cleaned2002.csv',
    '2003': './data/pp-cleaned2003.csv',
    '2004': './data/pp-cleaned2004.csv',
    '2005': './data/pp-cleaned2005.csv',
    '2006': './data/pp-cleaned2006.csv',
    '2007': './data/pp-cleaned2007.csv',
    '2008': './data/pp-cleaned2008.csv',
    '2009': './data/pp-cleaned2009.csv',
    '2010': './data/pp-cleaned2010.csv',
    '2011': './data/pp-cleaned2011.csv',
    '2012': './data/pp-cleaned2012.csv',
    '2013': './data/pp-cleaned2013.csv',
    '2014': './data/pp-cleaned2014.csv',
    '2015': './data/pp-cleaned2015.csv',
    '2016': './data/pp-cleaned2016.csv',
    '2017': './data/pp-cleaned2017.csv',
    '2018': './data/pp-cleaned2018.csv',
    '2019': './data/pp-cleaned2019.csv',
    '2020': './data/pp-cleaned2020.csv',
    '2021': './data/pp-cleaned2021.csv'
}

# Initialize an empty dataframe to hold all combined data
combined_data = pd.DataFrame()

# Process each file
for year, path in file_paths.items():
    # Load the data
    data = pd.read_csv(path)
    
    # Extract postcode prefix
    data['Postcode Prefix'] = data['Postcode'].apply(lambda x: x.split(' ')[0])
    
    # Add the year column to distinguish entries from different years
    data['Year'] = year
    
    # Append to the combined dataframe
    combined_data = combined_data._append(data, ignore_index=True)

# Now, combined_data contains all entries from 2019, 2020, and 2021 with postcode prefixes extracted
# Next, we'll group by postcode prefix and save to CSVs

# Unique list of postcode prefixes
postcode_prefixes_combined = combined_data['Postcode Prefix'].unique()

# Dictionary to store file paths for each combined prefix
saved_files_combined = {}

for prefix in postcode_prefixes_combined:
    # Filter data for the current prefix
    filtered_data = combined_data[combined_data['Postcode Prefix'] == prefix]
    
    # Define file name based on postcode prefix
    save_file_path_combined = f'/data/{prefix}_combined.csv'
    
    # Save to CSV
    filtered_data.to_csv(save_file_path_combined, index=False)
    
    # Store the file path
    saved_files_combined[prefix] = save_file_path_combined

# Display a sample of the saved file paths for combined data
list(saved_files_combined.items())[:5]
    