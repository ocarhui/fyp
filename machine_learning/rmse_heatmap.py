import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the ARIMA metrics CSV file
arima_metrics = pd.read_csv('ensemble_metrics.csv')


# Assume that 'RMSE' and 'borough' are the columns in your CSV file
rmse_values = arima_metrics['ensemble MAE'].values
borough_names = arima_metrics['borough'].values

# Define the color range to be more dramatic by focusing on the interquartile range
# This will enhance the color contrast for the middle 50% of your data
vmin, vmax = np.percentile(rmse_values, [25, 75])

# Create a color map from blue (cool) to red (warm)
cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])

# Normalize the RMSE values within the interquartile range for color mapping
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Create a grid for the heatmap
grid_size = int(np.ceil(np.sqrt(len(rmse_values))))  # Create as square a grid as possible
rmse_grid = np.full((grid_size**2), np.nan)  # Initialize with NaNs
rmse_grid[:len(rmse_values)] = rmse_values  # Fill in the RMSE values
rmse_grid = rmse_grid.reshape((grid_size, grid_size))  # Reshape to a square grid

# Plot the heatmap
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(rmse_grid, cmap=cmap, norm=norm)
ax.axis('off')  # Turn off the axis

# Add borough names on top of the squares
for (i, j), val in np.ndenumerate(rmse_grid):
    if not np.isnan(val):
        idx = i * grid_size + j
        if idx < len(borough_names):
            ax.text(j, i, borough_names[idx], ha='center', va='center', fontsize=8, color='white')

# Add a color bar with the scale
cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('MAE Value')

# Display the plot
plt.show()
