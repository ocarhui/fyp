import pandas as pd
import os

DATA_DIR = "./data"
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
        df.to_csv(file_path, index=False)

# Call the method to sort the CSV files
sort_csv_files_by_date()