import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preparation import preprocess_data, preprocess_data_without_arima
import optuna

DATA_DIR = "./data"
ARIMA_DIR = "./arima_only_predictions"

def load_data(data_dir):
    borough_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            borough_data[borough] = df
    return borough_data

def load_arima_only_predictions(data_dir):
    arima_only_predictions = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            borough = filename.split('_')[0]
            df = pd.read_csv(file_path)
            arima_only_predictions[borough] = df
    return arima_only_predictions

def build_model(shapeIn):
    model = keras.Sequential()
    model.add(layers.Dense(128, kernel_initializer='he_normal', input_shape=(shapeIn,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, kernel_initializer='he_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, kernel_initializer='he_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, kernel_initializer='he_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, kernel_initializer='he_normal'))

    opt = keras.optimizers.Adam(learning_rate=0.05, decay=1e-6)

    model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'root_mean_squared_error'])
    
    return model

def train_model(X_train, y_train, epochs=100, validation_split=0.2):
    input_shape = X_train.shape[1]
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    model = build_model(input_shape)
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=1, callbacks=[reduce_lr])
    return model, history

def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss, Test MAE, Test RMSE: {results}")
    return results

borough_data = load_data(DATA_DIR)
integrated_data = pd.read_csv(DATA_DIR+'/other_data_2/integrated.csv')
arima_data = load_arima_only_predictions(ARIMA_DIR)
#borough = "SW8"

for borough in borough_data:
#if borough in borough_data:
    print(f"Processing {borough} data...")
    borough_df = borough_data[borough].copy()
    X, y = borough_df.drop('Price', axis=1), borough_df['Price']
    
    # Assuming preprocess_data function adjusts for TensorFlow compatibility
    X_train, X_test, y_train, y_test, scaler, postcode_mapping = preprocess_data(borough, borough_df)
    
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')


    model, history = train_model(X_train, y_train, epochs=100, validation_split=0.2)
    
    #evaluate_model(model, X_test, y_test)
    postcode_query = next(iter(postcode_mapping))
    ptal_mapping = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}

    
    avg_prices = []

    for month in range(1,13):
        features_list = []

        for postcode in postcode_mapping:
            current_df = borough_df[borough_df['Postcode'] == postcode_mapping[postcode]]

            features = {
                'Postcode': postcode_mapping[postcode],
                'HouseSize': borough_df['HouseSize'].mean(),
                'EnergyEfficiency': borough_df['EnergyEfficiency'].mean(),
                'BuildDate': borough_df['BuildDate'].mean(),
                'Distance to station': current_df['Distance to station'].values[0],
                'Average Income': current_df['Average Income'].values[0],
                'IMD decile': current_df['IMD decile'].values[0],
                'NumOfRms': borough_df['NumOfRms'].mean(),
                'ARIMA_Predictions': arima_data[borough]['Difference'][month-1],
                'Month': month,
                'PTAL2021': current_df['PTAL2021'].values[0],
                'London zone': current_df['London zone'].values[0]
            }

            features_list.append(features)
        features_df = pd.DataFrame(features_list)
        X_pred = features_df
        X_pred_scaled = scaler.transform(X_pred)
        predictions = make_predictions(model, X_pred_scaled)
        avg_price = np.mean(predictions)
        avg_prices.append(avg_price)

    avg_prices_df = pd.DataFrame(avg_prices, columns=['Average_Price'])
    avg_prices_df['Month'] = range(1,13)
    avg_prices_df.to_csv(f'./tf_predictions/{borough}_predicted.csv', index=False)     
            



