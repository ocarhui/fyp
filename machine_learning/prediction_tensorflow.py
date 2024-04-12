import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preparation import preprocess_data, preprocess_data_without_arima, preprocess_data_with_date
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

def save_results(results_str, file_path):
    with open(file_path, "w") as file:
        file.write(results_str)
    print(f"Results have been saved to {file_path}.")

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

def build_model_from_csv(shapeIn, params_df, borough):
    # Select the row corresponding to the specified borough
    row = params_df.loc[params_df['Borough'] == borough].iloc[0]
    
    # Initialize the model
    model = keras.Sequential()
    
    # Add layers as specified in the CSV file
    for i in range(int(row['n_layers'])):
        # Determine the number of units for this layer
        n_units = row[f'n_units_l{i}']
        if pd.notna(n_units):  # Check if n_units is not NaN
            model.add(layers.Dense(int(n_units), kernel_initializer='he_normal'))
            model.add(layers.Activation('relu'))
            if pd.notna(row['dropout_rate']):  # Check if dropout_rate is not NaN
                model.add(layers.Dropout(row['dropout_rate']))
    
    # Add the output layer
    model.add(layers.Dense(1, kernel_initializer='he_normal'))
    
    # Configure the optimizer with parameters from the CSV
    opt = keras.optimizers.Adam(learning_rate=row['lr'], decay=1e-6)
    
    # Compile the model
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

def train_model_with_params(X_train, y_train, param_df, borough, epochs=100, validation_split=0.2):
    input_shape = X_train.shape[1]
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    model = build_model_from_csv(input_shape, param_df, borough)
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=1, callbacks=[reduce_lr])
    return model, history

def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss, Test MAE, Test RMSE: {results}")
    return results

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Example: Log-uniform distribution
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)  # Example: Uniform distribution
    n_layers = trial.suggest_int('n_layers', 1, 3)  # Number of layers
    
    # Dynamic definition of the model based on Optuna's suggestions
    model = keras.Sequential()
    for i in range(n_layers):
        num_neurons = trial.suggest_int(f'n_units_l{i}', 16, 128, log=True)
        model.add(layers.Dense(num_neurons, activation='relu'))
        model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile model with the suggested learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error')
    
    # Fit the model (consider using a subset of data to speed up the process)
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
    
    # Evaluate the model on the validation set
    val_loss = model.evaluate(X_test, y_test, verbose=0)
    
    # Objective: Minimize validation loss
    return val_loss

def plot_predictions(historical_data, predictions, borough_name):
    """
    Plots historical prices along with the predictions and saves the plot.

    :param historical_data: DataFrame with columns 'Date' and 'Price' for historical data
    :param predictions: DataFrame with columns 'Date' and 'Predicted_Price' for predictions
    :param borough_name: The name of the borough for title and filename
    """
    plt.figure(figsize=(12, 6))

    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    monthly_avg_price = historical_data.resample('M', on='Date')['Price'].mean().reset_index()
    monthly_avg_price['Price_Diff'] = monthly_avg_price['Price'].diff().dropna()
    monthly_avg_price_cleaned = monthly_avg_price.dropna(subset=['Price_Diff']) 

    monthly_means = monthly_avg_price_cleaned.groupby(monthly_avg_price_cleaned['Date'].dt.to_period('M')).mean()
    monthly_means.index = monthly_means.index.to_timestamp()
    
    # Plot historical data
    if monthly_avg_price is not None and not monthly_avg_price.empty:
        plt.plot(monthly_means.index, monthly_means['Price'], label='Historical Prices', color='blue')
    
    # Plot predictions
    plt.plot(predictions['Date'], predictions['Average_Price'], color='red', label='Predicted Prices')
    plt.title(f'Price Predictions for {borough_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'./tf_predictions/{borough_name}_price_predictions.png')

borough_data = load_data(DATA_DIR)
integrated_data = pd.read_csv(DATA_DIR+'/other_data_2/integrated.csv')
arima_data = load_arima_only_predictions(ARIMA_DIR)
tf_params = pd.read_csv("tf_params.csv")

for borough in borough_data:

    borough_df = borough_data[borough].copy()
    X_train, X_test, y_train, y_test, scaler, postcode_mapping = preprocess_data_with_date(borough, borough_df)
    print(f"processing {borough}")
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    model, history = train_model_with_params(X_train, y_train, tf_params, borough, epochs=100, validation_split=0.2)
    
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
                'Timestamp': pd.to_datetime(f'2022-{month}-01'),
                'PTAL2021': current_df['PTAL2021'].values[0],
                'London zone': current_df['London zone'].values[0]
            }

            

        features_list.append(features)
        features_df = pd.DataFrame(features_list)
        features_df['Timestamp'] = features_df['Timestamp'].astype('int64') // 10**9
        X_pred = features_df
        X_pred_scaled = scaler.transform(X_pred)
        predictions = make_predictions(model, X_pred_scaled)
        avg_price = np.mean(predictions)
        avg_prices.append(avg_price)

    results = evaluate_model(model, X_test, y_test)

    avg_prices_df = pd.DataFrame(avg_prices, columns=['Average_Price'])
    last_date = borough_df['Date'].iloc[-1]
    forecast = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq='M')
    forecast = pd.DataFrame(forecast, columns=['Date'])
    forecast['Average_Price'] = avg_prices
    avg_prices_df.to_csv(f'./tf_predictions/{borough}_predicted.csv', index=False)
    save_results(str(results), f"./tf_predictions/{borough}_analyse_results.txt")

    plot_predictions(borough_df, forecast, borough)
            



