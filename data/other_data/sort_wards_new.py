import pandas as pd

data = pd.read_csv('./data/other_data/wards.csv', encoding='ISO-8859-1')  # Assuming ISO-8859-1 encoding; adjust if necessary

# List of corrected postal code prefixes
postcode_prefixes = ["Barking and Dagenham", "Barnet", "Bexley", 
                     "Brent", "Bromley", "Camden", "Croydon", "Ealing", 
                     "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham", 
                     "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", 
                     "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", 
                     "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", 
                     "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster", "City of London"]

# Filter the data
filtered_data = data[data['lad11nm'].apply(lambda x: any(x.strip().startswith(prefix) for prefix in postcode_prefixes))]
filtered_data = filtered_data.drop(["pcd7", "pcd8", "par11cd", "par11nm", "par11nmw", "wd11nm", "wd11nmw", "lad11nm", "lad11nmw"], axis=1)

# Save the filtered data to a new CSV file
filtered_data.to_csv('./data/other_data/filtered_postcodes.csv', index=False)