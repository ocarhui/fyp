import pandas as pd
DATA_DIR = "./data"

# Specify the columns to check for empty values
columns_to_check = ['priceper', 'CONSTRUCTION_AGE_BAND']

# Loop through each year from 1995 to 2021
for year in range(1995, 2022):
    # Define the file path for the current year
    file_path = f"{DATA_DIR}/pp-sorted{year}.csv"
    
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Remove rows with any empty values in the specified columns
    df_cleaned = df.dropna(subset=columns_to_check)

    # Define the cleaned file path for the current year
    cleaned_file_path = f"{DATA_DIR}/pp-cleaned{year}.csv"
    
    # Save the cleaned DataFrame back to a new CSV file
    df_cleaned.to_csv(cleaned_file_path, index=False)