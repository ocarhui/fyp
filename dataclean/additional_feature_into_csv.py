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

def merge_dfs_on_postcode(borough_df, transport_df, postcode_col='Postcode'):
    merged_df = pd.merge(borough_df, transport_df, how='left', left_on='Postcode_Short', right_on='Postcode_Long', suffixes=('', '_Short'))
    merged_df = pd.merge(borough_df, transport_df, how='left', left_on='Postcode_Medium', right_on='Postcode_Long', suffixes=('', '_Medium'))
    merged_df = pd.merge(borough_df, transport_df, how='left', on='Postcode_Long')

    return merged_df

DATA_DIR = "./data"
borough_data = load_data(DATA_DIR)

# Load transport IMD data and normalize with expanded postcodes
transport_imd_df = pd.read_csv('./data/other_data_2/integrated.csv')
transport_imd_df.drop_duplicates(subset='Postcode', keep="last", inplace=True)
expand_postcodes_for_merge(transport_imd_df)
full_postcode = transport_imd_df.copy()
full_postcode.drop(['Postcode_Medium', 'Postcode_Short'], axis=1)
full_postcode_dict = full_postcode.set_index('Postcode').to_dict(orient='index')
medium_postcode = transport_imd_df.drop_duplicates(subset='Postcode_Medium', keep="last")
medium_postcode.drop(['Postcode_Long', 'Postcode_Short'], axis=1, inplace=True)
medium_postcode_dict = medium_postcode.set_index('Postcode_Medium').to_dict(orient='index')
short_posecode = transport_imd_df.drop_duplicates(subset='Postcode_Short', keep="last")
short_posecode.drop(['Postcode_Long', 'Postcode_Medium'], axis=1, inplace=True)
short_postcode_dict = short_posecode.set_index('Postcode_Short').to_dict(orient='index')



for borough in borough_data:
    borough_df = borough_data[borough].copy()

    expand_postcodes_for_merge(borough_df)
    borough_df["Index of Multiple Deprivation"] = None
    borough_df["London zone"] = None
    borough_df["Nearest station"] = None
    borough_df["Distance to station"] = None
    borough_df["Average Income"] = None
    borough_df["IMD decile"] = None
    borough_df["PTAL2021"] = None
    borough_df["AI2021"] = None

    
    for row in borough_df.iterrows():
        row = row[1]
        full_postcode = row['Postcode_Long']
        medium_postcode = row['Postcode_Medium']
        short_postcode = row['Postcode_Short']
        
        if full_postcode in full_postcode_dict:
            borough_df.at[row.name, "Index of Multiple Deprivation"] = full_postcode_dict[full_postcode]['Index of Multiple Deprivation']
            borough_df.at[row.name, "London zone"] = full_postcode_dict[full_postcode]['London zone']
            borough_df.at[row.name, "Nearest station"] = full_postcode_dict[full_postcode]['Nearest station']
            borough_df.at[row.name, "Distance to station"] = full_postcode_dict[full_postcode]['Distance to station']
            borough_df.at[row.name, "Average Income"] = full_postcode_dict[full_postcode]['Average Income']
            borough_df.at[row.name, "IMD decile"] = full_postcode_dict[full_postcode]['IMD decile']
            borough_df.at[row.name, "PTAL2021"] = full_postcode_dict[full_postcode]['PTAL2021']
            borough_df.at[row.name, "AI2021"] = full_postcode_dict[full_postcode]['AI2021']
        elif medium_postcode in medium_postcode_dict:
            borough_df.at[row.name, "Index of Multiple Deprivation"] = medium_postcode_dict[medium_postcode]['Index of Multiple Deprivation']
            borough_df.at[row.name, "London zone"] = medium_postcode_dict[medium_postcode]['London zone']
            borough_df.at[row.name, "Nearest station"] = medium_postcode_dict[medium_postcode]['Nearest station']
            borough_df.at[row.name, "Distance to station"] = medium_postcode_dict[medium_postcode]['Distance to station']
            borough_df.at[row.name, "Average Income"] = medium_postcode_dict[medium_postcode]['Average Income']
            borough_df.at[row.name, "IMD decile"] = medium_postcode_dict[medium_postcode]['IMD decile']
            borough_df.at[row.name, "PTAL2021"] = medium_postcode_dict[medium_postcode]['PTAL2021']
            borough_df.at[row.name, "AI2021"] = medium_postcode_dict[medium_postcode]['AI2021']
        elif short_postcode in short_postcode_dict:
            borough_df.at[row.name, "Index of Multiple Deprivation"] = short_postcode_dict[short_postcode]['Index of Multiple Deprivation']
            borough_df.at[row.name, "London zone"] = short_postcode_dict[short_postcode]['London zone']
            borough_df.at[row.name, "Nearest station"] = short_postcode_dict[short_postcode]['Nearest station']
            borough_df.at[row.name, "Distance to station"] = short_postcode_dict[short_postcode]['Distance to station']
            borough_df.at[row.name, "Average Income"] = short_postcode_dict[short_postcode]['Average Income']
            borough_df.at[row.name, "IMD decile"] = short_postcode_dict[short_postcode]['IMD decile']
            borough_df.at[row.name, "PTAL2021"] = short_postcode_dict[short_postcode]['PTAL2021']
            borough_df.at[row.name, "AI2021"] = short_postcode_dict[short_postcode]['AI2021']

    borough_df.drop(['Postcode_Long', 'Postcode_Medium', 'Postcode_Short'], axis=1, inplace=True)
    print(borough+" "+str(borough_df['PTAL2021'].isnull().sum()))
    print("LZ:"+" "+borough+str(borough_df['London zone'].isnull().sum()))
            
    # Save the merged DataFrame to a new CSV file
    final_file_path = f'./data/{borough}_combined.csv'
    borough_df.to_csv(final_file_path, index=False)
