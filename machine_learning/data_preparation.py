import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_data(borough, data):
    """
    Prepares the dataset for training by handling missing values, encoding categorical and ordinal variables,
    scaling numerical features, and preparing the dataset for training.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    
    data['NumOfRms'] = imputer.fit_transform(data[['NumOfRms']])
    data['ARIMA_Predictions'] = imputer.fit_transform(data[['ARIMA_Predictions']])
    
    # Convert 'Date' to datetime and extract 'Month' and 'Year'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    postcode_mapping = {postcode: i for i, postcode in enumerate(data['Postcode'].unique())}
    data['Postcode'] = data['Postcode'].map(postcode_mapping)
    
    # Drop 'Type' and 'Ownership'
    data.drop(columns=['Type', 'Ownership'], inplace=True)
    data.fillna(0, inplace=True)
    
    # Encode 'PTAL2021' based on the specified categorization
    ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
    data['PTAL2021'] = data['PTAL2021'].map(ptal_mapping)
    
    # Assuming 'London zone' is already in a suitable format for ordinal encoding
    
    # Select features for training
    features_to_use = ['Postcode', 'HouseSize', 'EnergyEfficiency', 'BuildDate', 'Distance to station',
                       'Average Income', 'IMD decile', 'NumOfRms', 'ARIMA_Predictions', 'Month', 'PTAL2021', 'London zone']
    features = data[features_to_use]
    target = data['Price']
    
    # Scale features
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # If you're scaling features, do it here but ensure you fit the scaler on the training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, postcode_mapping

def preprocess_data_without_arima(borough, data):
    """
    Prepares the dataset for training by handling missing values, encoding categorical and ordinal variables,
    scaling numerical features, and preparing the dataset for training.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    
    data['NumOfRms'] = imputer.fit_transform(data[['NumOfRms']])
    # Convert 'Date' to datetime and extract 'Month' and 'Year'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    postcode_mapping = {postcode: i for i, postcode in enumerate(data['Postcode'].unique())}
    data['Postcode'] = data['Postcode'].map(postcode_mapping)
    
    # Drop 'Type' and 'Ownership'
    data.drop(columns=['Type', 'Ownership'], inplace=True)
    data.fillna(0, inplace=True)
    
    # Encode 'PTAL2021' based on the specified categorization
    ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
    data['PTAL2021'] = data['PTAL2021'].map(ptal_mapping)
    
    # Assuming 'London zone' is already in a suitable format for ordinal encoding
    
    # Select features for training
    features_to_use = ['Postcode', 'HouseSize', 'EnergyEfficiency', 'BuildDate', 'Distance to station',
                       'Average Income', 'IMD decile', 'NumOfRms', 'Month', 'PTAL2021', 'London zone']
    features = data[features_to_use]
    target = data['Price']
    
    # Scale features
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # If you're scaling features, do it here but ensure you fit the scaler on the training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, postcode_mapping