import pandas as pd
def process_and_save_by_postcode_prefix(file_path, year):

    data = pd.read_csv(file_path)
    # Extract postcode prefix
    data['Postcode Prefix'] = data['Postcode'].apply(lambda x: x.split(' ')[0])
    
    # Unique list of postcode prefixes
    postcode_prefixes = data['Postcode Prefix'].unique()
    
    # Dictionary to store file paths for each prefix
    saved_files = {}
    
    for prefix in postcode_prefixes:
        # Filter data for the current prefix
        filtered_data = data[data['Postcode Prefix'] == prefix]
        
        # Define file name based on postcode prefix and year
        save_file_path = f'./data/{prefix}_{year}.csv'
        
        # Save to CSV
        filtered_data.to_csv(save_file_path, index=False)
        
        # Store the file path
        saved_files[prefix] = save_file_path
    
    return saved_files

   
DATA_DIR = "./data"
DATA_1995_CSV = DATA_DIR + "/pp-cleaned1995.csv"
DATA_1996_CSV = DATA_DIR + "/pp-cleaned1996.csv"
DATA_1997_CSV = DATA_DIR + "/pp-cleaned1997.csv"
DATA_1998_CSV = DATA_DIR + "/pp-cleaned1998.csv"
DATA_1999_CSV = DATA_DIR + "/pp-cleaned1999.csv"
DATA_2000_CSV = DATA_DIR + "/pp-cleaned2000.csv"
DATA_2001_CSV = DATA_DIR + "/pp-cleaned2001.csv"
DATA_2002_CSV = DATA_DIR + "/pp-cleaned2002.csv"
DATA_2003_CSV = DATA_DIR + "/pp-cleaned2003.csv"
DATA_2004_CSV = DATA_DIR + "/pp-cleaned2004.csv"
DATA_2005_CSV = DATA_DIR + "/pp-cleaned2005.csv"
DATA_2006_CSV = DATA_DIR + "/pp-cleaned2006.csv"
DATA_2007_CSV = DATA_DIR + "/pp-cleaned2007.csv"
DATA_2008_CSV = DATA_DIR + "/pp-cleaned2008.csv"
DATA_2009_CSV = DATA_DIR + "/pp-cleaned2009.csv"
DATA_2010_CSV = DATA_DIR + "/pp-cleaned2010.csv"
DATA_2011_CSV = DATA_DIR + "/pp-cleaned2011.csv"
DATA_2012_CSV = DATA_DIR + "/pp-cleaned2012.csv"
DATA_2013_CSV = DATA_DIR + "/pp-cleaned2013.csv"
DATA_2014_CSV = DATA_DIR + "/pp-cleaned2014.csv"
DATA_2015_CSV = DATA_DIR + "/pp-cleaned2015.csv"
DATA_2016_CSV = DATA_DIR + "/pp-cleaned2016.csv"
DATA_2017_CSV = DATA_DIR + "/pp-cleaned2017.csv"
DATA_2018_CSV = DATA_DIR + "/pp-cleaned2018.csv"
DATA_2019_CSV = DATA_DIR + "/pp-cleaned2019.csv"
DATA_2020_CSV = DATA_DIR + "/pp-cleaned2020.csv"
DATA_2021_CSV = DATA_DIR + "/pp-cleaned2021.csv"

year = 1995

# Process and save files for 2019
dataList = [DATA_1995_CSV, DATA_1996_CSV, DATA_1997_CSV, DATA_1998_CSV, DATA_1999_CSV, DATA_2000_CSV, DATA_2001_CSV,
            DATA_2002_CSV, DATA_2003_CSV, DATA_2004_CSV, DATA_2005_CSV, DATA_2006_CSV, DATA_2007_CSV, DATA_2008_CSV,
            DATA_2009_CSV, DATA_2010_CSV, DATA_2011_CSV, DATA_2012_CSV, DATA_2013_CSV, DATA_2014_CSV, DATA_2015_CSV,
            DATA_2016_CSV, DATA_2017_CSV, DATA_2018_CSV, DATA_2019_CSV, DATA_2020_CSV, DATA_2021_CSV]

for data in dataList:
    saved_files = process_and_save_by_postcode_prefix(data, str(year))
    year += 1
    