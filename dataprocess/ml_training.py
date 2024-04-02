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


for borough in borough_data:
    print(borough)
"""data = df_with_arima("SW11", borough_data["SW11"], arima_params)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Calculate the age of the property at the time of sale
data['PropertyAge'] = data['Year'] - data['BuildDate']

# Define columns to be included in preprocessing
categorical_cols = ['Postcode', 'Town', 'City', 'County']
numerical_cols = ['PricePer', 'NumOfRms', 'HouseSize', 'EnergyEfficiency', 'PropertyAge']

# Define transformers for the preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Apply preprocessing
X = data#.drop(['Price', 'Date', 'ARIMA_Predictions', 'ID', 'Street Number', 'Flat Number', 'Street Name', 'Area', 'BuildDate', 'Year', 'Month'], axis=1)
y = data['Price']

X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print(mae_xgb, mse_xgb, rmse_xgb)

"""