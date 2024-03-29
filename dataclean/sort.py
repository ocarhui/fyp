import pandas as pd
import os

DATA_DIR = "./data"
def convert_build_date(build_date):
    """Convert build date ranges to mean year or handle specific text cases."""
    if ':' in build_date:
        if 'before' in build_date:
            return 1900  # Default year for "before 1900"
        years = build_date.split(':')[1].split('-')
        years = [year for year in years if year.isdigit()]
        if years:
            mean_year = sum(int(year) for year in years) / len(years)
            return round(mean_year)
    return build_date

def sort_csv_files_by_date():
    # Get the list of CSV files in the data directory
    csv_files = [file for file in os.listdir(DATA_DIR) if file.endswith(".csv")]

    # Sort the CSV files by modification date
    sorted_csv_files = sorted(csv_files, key=lambda file: os.path.getmtime(os.path.join(DATA_DIR, file)))

    # Read each CSV file, sort by date column, and overwrite the original file
    for csv_file in sorted_csv_files:
        file_path = os.path.join(DATA_DIR, csv_file)
        df = pd.read_csv(file_path)
        df.sort_values(by="Date", inplace=True)  # Replace "date" with the actual column name
        df['BuildDate'] = df['BuildDate'].apply(convert_build_date)
        df['Newbuild'] = df['Newbuild'].map({'N': 0, 'Y': 1})

        # Columns to drop
        columns_to_drop = [
            'priceper', 'year', 'dateoftransfer', 'propertytype', 'duration', 'price', 'postcode', 
            'lad21cd', 'transactionid', 'id', 'tfarea', 'numberrooms', 'classt', 
            'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY', 'CONSTRUCTION_AGE_BAND'
        ]
        
        df = df.drop(columns=columns_to_drop, errors='ignore')
        df.to_csv(file_path, index=False)

# Call the method to sort the CSV files
sort_csv_files_by_date()