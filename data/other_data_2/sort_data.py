import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the datasets
transport_index_path = './data/other_data_2/TFL_PTAL.csv'  # Update this path
filtered_postcodes_path = './data/other_data_2/London_postcodes.csv'  # Update this path

transport_index_df = pd.read_csv(transport_index_path)
filtered_postcodes_df = pd.read_csv(filtered_postcodes_path)
filtered_postcodes_df = filtered_postcodes_df[['Postcode', 'Ward Code', 'Northing', 'Easting', 'London zone', 'Index of Multiple Deprivation', 'Nearest station', 'Distance to station', 'Average Income', 'IMD decile']]
filtered_postcodes_df['key'] = 1
transport_index_df['key'] = 1

gdf_postcodes = gpd.GeoDataFrame(
    filtered_postcodes_df,
    geometry=[Point(xy) for xy in zip(filtered_postcodes_df.Easting, filtered_postcodes_df.Northing)]
)

gdf_transport = gpd.GeoDataFrame(
    transport_index_df,
    geometry=[Point(xy) for xy in zip(transport_index_df.X, transport_index_df.Y)]
)

# Set CRS (Coordinate Reference System) to something appropriate, e.g., British National Grid
gdf_postcodes.crs = "EPSG:27700"
gdf_transport.crs = "EPSG:27700"

# Buffer the points in gdf_postcodes by 50 units (the tolerance)
# Adjust this value based on the units of your CRS if necessary
gdf_postcodes['geometry'] = gdf_postcodes.geometry.buffer(70)

# Perform spatial join
# This joins gdf_transport to gdf_postcodes where their geometries (including buffers) intersect
merged_gdf = gpd.sjoin(gdf_postcodes, gdf_transport, how="inner", op="intersects")

# Convert back to DataFrame if necessary
merged_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))

"""def find_matches(row, df2, easting_tolerance=50, northing_tolerance=50):
    condition = (df2['X'].sub(row['Easting']).abs() <= easting_tolerance) & \
                (df2['Y'].sub(row['Northing']).abs() <= northing_tolerance)
    matches = df2[condition]
    # Add the current row's data to the matches, simulating a 'merge'
    for column in row.index:
        matches[column + '_x'] = row[column]
    return matches

# Perform the cross join
results = [find_matches(row, transport_index_df) for index, row in filtered_postcodes_df.iterrows()]

# Concatenate all matches into a single DataFrame
merged_df = pd.concat(results, ignore_index=True)"""

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('./data/other_data_2/integrated.csv', index=False)