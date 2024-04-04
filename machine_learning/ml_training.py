import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from arima_in_sample_predictions import df_with_arima
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from xgboost import XGBRegressor

DATA_DIR = "./data"

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

borough_data = load_data(DATA_DIR)
arima_file_path = 'arima_params.csv'
df = pd.read_csv(arima_file_path)
arima_params = df.set_index('Borough').to_dict()['ARIMA_Params']

#borough = "SW11"
average_price_prediction_2022 = {}

for borough in borough_data:
#if borough in borough_data:
    
    data = borough_data[borough].copy()

    #data.drop(columns=['Newbuild'], inplace=True)

    # Now, let's handle missing values.
    # For 'Flat Number' and 'NumOfRms', since they are categorical and numerical respectively, 
    # we can fill in missing values with a placeholder for 'Flat Number' and the median for 'NumOfRms'.
    data['Flat Number'].fillna('None', inplace=True)
    data['NumOfRms'].fillna(data['NumOfRms'].median(), inplace=True)

    # For 'Area', due to a large number of missing values, it's best to fill in with a placeholder as well.
    data['Area'].fillna('Unknown', inplace=True)

    # Check for any remaining missing values
    missing_values = data.isnull().sum()

    # Encoding categorical variables: Let's identify the categorical columns first.
    categorical_columns = data.select_dtypes(include=['object']).columns

    # We will use one-hot encoding for categorical variables.
    # To avoid increasing dimensionality too much, let's focus on encoding the 'Type' and 'Ownership' columns,
    # as they seem to be directly related to the house pricing.
    data_encoded = pd.get_dummies(data, columns=['Type', 'Ownership'], drop_first=True)

    # Now, let's create a new feature 'Age' which is derived from the 'BuildDate' column.
    # Assuming the dataset was last updated in 2021, we'll calculate the age of the house until 2021.
    data_encoded['Age'] = 2021 - data_encoded['BuildDate']
    data_encoded.loc[data_encoded['BuildDate'] == 0, 'Age'] = 0  # For new builds, age should be 0

    # Remove the 'Date' column as we already have the 'Year' column.
    data_encoded.drop(columns=['Date'], inplace=True)
    print(borough)

    # Finally, let's convert the 'IMD Extent %' from string to float by removing the percentage sign and dividing by 100.
    data_encoded['IMD Extent %'] = data_encoded['IMD Extent %'].str.rstrip('%').astype('float') / 100

    # Let's check the dataset again after encoding and feature engineering
    data_encoded_info = data_encoded.info()
    data_encoded_head = data_encoded.head()
    #print(missing_values, categorical_columns, data_encoded_info, data_encoded_head)

    X = data_encoded.drop(columns=['ID', 'Price', 'Postcode', 'Street Number', 'Flat Number', 'Street Name', 
                               'Area', 'Town', 'City', 'County', 'Postcode Prefix','Type_S', 'Type_F', 'Ownership_L', 'Type_T'])
    
    y = data_encoded['Price']

    # Split the data into training and validation sets
    # We'll use 80% of the data for training and 20% for validation.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the shape of the created data splits
    (X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    xgb_model = XGBRegressor(random_state=42)

# Fit the model to the training data
    xgb_model.fit(X_train, y_train)

    # Predict the prices on the validation set
    y_pred = xgb_model.predict(X_val)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(mae, rmse)

    data_2021 = data_encoded[data_encoded['Year'] == 2021]

    # Calculate the mean of the features for 2021, excluding 'Price' which is what we want to predict
    # Also exclude 'Year' since we're predicting for 2022
    numeric_cols = data_2021.select_dtypes(include=[np.number]).columns.tolist()
    features_mean_2021 = data_2021[numeric_cols].drop(columns=['Price']).mean().to_frame().transpose()

    # We should set 'Year' to 2022 since we're using this data to predict for 2022
    features_mean_2021['Year'] = 2022

    borough_prediction = xgb_model.predict(features_mean_2021)

    average_price_prediction_2022[borough] = borough_prediction[0]

    print(average_price_prediction_2022)