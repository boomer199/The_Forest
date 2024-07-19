# THIS FILE IS RUN JUST TO TEST ACCURACY AND LOG PREVIOUS PREDICTIONS TO A JSON THEN EVAL ACCURACY
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import csv
import json
import ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from datetime import datetime, timedelta, date

def grab_price_data():

    # Define the list of tickers
    tickers_list = ['JPM', 'COST', 'IBM', 'HD', 'ARWR'] #JPMorgan, Costco, IBM, Home Depot, Arrowhead Pharmaceuticals
    
    # I need to store multiple result sets.
    full_price_history = []

    for ticker in tickers_list:

        # Grab the daily price history for 2 years
        stock_data = yf.download(ticker, period='2y', interval='1d')

        # Reset index to make 'Date' a column instead of the index
        stock_data.reset_index(inplace=True)

        # Rename the 'Date' column to 'datetime' to match the expected column names
        stock_data.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        # Add a 'symbol' column to the dataframe
        stock_data['symbol'] = ticker

        # Append the dataframe to the list
        full_price_history.append(stock_data)

    # Concatenate all dataframes in the list into a single dataframe
    all_data = pd.concat(full_price_history)

    # Dump the data to a CSV file, don't have an index column
    all_data.to_csv('price_data.csv', index=False)

# Run the function to grab the data
grab_price_data()

if os.path.exists('price_data.csv'):
    # Load the data
    price_data = pd.read_csv('price_data.csv')
else:
    # Grab the data and store it.
    grab_price_data()
    # Load the data
    price_data = pd.read_csv('price_data.csv')


price_data = price_data[["symbol", "datetime", "close", "high", "low", "open", "volume"]]

#Sort the data by symbol and datetime
price_data.sort_values(by=['symbol', 'datetime'], inplace=True) #inplace=true means that the changes are saved to the dataframe and we dont need another variable to store the changes

price_data["change_in_price"] = price_data["close"].diff()

#identify the rows where the ticker symbol changes, and then shift the change_in_price value to NaN
mask = price_data["symbol"] != price_data["symbol"].shift(1)

#change change_in_price to NaN where needed
price_data["change_in_price"] = np.where(mask == True, np.nan, price_data["change_in_price"]) #this is a numpy function that takes 3 arguments, the first is the condition, the second is the value to be used if the
                                                                                              #condition is true, and the third is the value to be used if the condition is false
                                                                                              
#see all null vals                                                                                              
price_data[price_data.isna().any(axis=1)]

def smooth_data(days_out):
    # Group by symbol, then apply the rolling function and grab the Min and Max.
    price_data_smoothed = price_data.groupby(['symbol'])[['close','low','high','open','volume']].transform(lambda x: x.ewm(span = days_out).mean())

    # Join the smoothed columns with the symbol and datetime column from the old data frame.
    smoothed_df = pd.concat([price_data[['symbol','datetime']], price_data_smoothed], axis=1, sort=False)
    
    # create a new column that will house the flag, and for each group calculate the diff compared to days_out days ago. Then use Numpy to define the sign.
    price_data['Signal_Flag'] = smoothed_df.groupby('symbol')['close'].transform(lambda x : np.sign(x.diff(days_out)))
smooth_data(5)

# Calculate the 14 day RSI
n = 14

# First make a copy of the data frame twice (exact same both times)
up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()

# For up days, if the change is less than 0 set to 0. This will only contain positive values.
up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

# For down days, if the change is greater than 0 set to 0. This will only contain negative values.
down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

# We need change in price to be absolute, so we use abs to make it so. (Only for down days)
down_df['change_in_price'] = down_df['change_in_price'].abs()

# Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean()) #The ewm function is exponential weighted moving average over a specified span of n days
ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean()) #The groupby makes it so the ewm is done by symbol and not over the whole dataset


# Calculate the Relative Strength
relative_strength = ewma_up / ewma_down

# Calculate the Relative Strength Index
relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

# Add the info to the data frame.
price_data['down_days'] = down_df['change_in_price']
price_data['up_days'] = up_df['change_in_price']
price_data['RSI'] = relative_strength_index

# Calculate the Stochastic Oscillator
n = 14

# Make a copy of the high and low column. (Instead of change in price like we did with RSI)
low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min()) #Pandas dataframe.rolling() function provides the feature of rolling window calculations.
high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

# Calculate the Stochastic Oscillator.
k_percent = 100 * ((price_data['close'] - low_14) / (high_14 - low_14))

# Add the info to the data frame.
price_data['low_14'] = low_14
price_data['high_14'] = high_14
price_data['k_percent'] = k_percent


# Calculate the Williams %R. Until r_percent is calculated, the code is IDENTICAL to the Stochastic Oscillator.
n = 14

# Make a copy of the high and low column.
low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

# Calculate William %R indicator.
r_percent = ((high_14 - price_data['close']) / (high_14 - low_14)) * - 100

# Add the info to the data frame. 
price_data['r_percent'] = r_percent

# Calculate the MACD
ema_26 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 26).mean())
ema_12 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 12).mean())
macd = ema_12 - ema_26

# Calculate the EMA
ema_9_macd = macd.ewm(span = 9).mean()

# Store the data in the data frame.
price_data['MACD'] = macd
price_data['MACD_EMA'] = ema_9_macd


# Calculate the Price Rate of Change
n = 9

# Calculate the Rate of Change in the Price, and store it in the Data Frame.
price_data['Price_Rate_Of_Change'] = price_data.groupby('symbol')['close'].transform(lambda x: x.pct_change(periods = n)) #pct_change computes the fractional change from the immediately previous row by default. 
                                                                                                                            #This is useful in comparing the fraction of change in a time series of elements.


def obv(group):
    """
    Calculate the On Balance Volume (OBV) for a given group of data.

    Parameters:
    - group (pandas.DataFrame): A pandas DataFrame containing the 'volume' and 'close' columns.

    Returns:
    - pandas.Series: A pandas Series containing the calculated OBV values.

    Example:
    >>> data = pd.DataFrame({'volume': [100, 200, 150, 300], 'close': [10, 12, 11, 13]})
    >>> obv(data)
    0    100
    1    300
    2    150
    3    450
    dtype: int64
    """
    # Grab the volume and close column.
    volume = group['volume']
    change = group['close'].diff()

    # intialize the previous OBV
    prev_obv = 0
    obv_values = []

    # calculate the On Balance Volume
    for i, j in zip(change, volume):

        if i > 0:
            current_obv = prev_obv + j
        elif i < 0:
            current_obv = prev_obv - j
        else:
            current_obv = prev_obv

        # OBV.append(current_OBV)
        prev_obv = current_obv
        obv_values.append(current_obv)
    
    # Return a panda series.
    return pd.Series(obv_values, index = group.index)
        

# apply the function to each group
obv_groups = price_data.groupby('symbol').apply(obv)

# add to the data frame, but drop the old index, before adding it.
price_data['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)

closed_groups = price_data.groupby('symbol')['close']

closed_groups = closed_groups.transform(lambda x: x.shift(1) < x)

price_data["Prediction"] = closed_groups * 1

# We need to remove all rows that have an NaN value.
print('Before NaN Drop we have {} rows and {} columns'.format(price_data.shape[0], price_data.shape[1]))

# Any row that has a `NaN` value will be dropped.
price_data = price_data.dropna()

# Display how much we have left now.
print('After NaN Drop we have {} rows and {} columns'.format(price_data.shape[0], price_data.shape[1]))

# Grab our X & Y Columns.
X_Cols = price_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume', 'Signal_Flag']]
Y_Cols = price_data['Prediction']

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)


# Define the best parameters found (later step)
best_params = {
    'subsample': 0.8,
    'reg_lambda': 2.5,
    'reg_alpha': 0.1,
    'n_estimators': 200,
    'max_depth': 2,
    'learning_rate': 0.05,
    'gamma': 0.3,
    'colsample_bytree': 0.8
}

# Create a Random Forest and Gradiant Boosting Classifier (using xgb boost)
xgb_clf = XGBClassifier(n_estimators=10, max_depth=2, learning_rate=1, objective='binary:logistic')
rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

# Fit the data to the model
xgb_clf.fit(X_train, y_train)
rand_frst_clf.fit(X_train, y_train)

# Make predictions
y_pred = rand_frst_clf.predict(X_test)
xgb_y_pred = xgb_clf.predict(X_test)

print('RF_Correct Prediction (%): ', accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)
print('XGB Correct Prediction (%): ', accuracy_score(y_test, xgb_clf.predict(X_test), normalize = True) * 100.0)


best_params = {
    'subsample': 0.8,
    'reg_lambda': 2.5,
    'reg_alpha': 0.1,
    'n_estimators': 200,
    'max_depth': 2,
    'learning_rate': 0.05,
    'gamma': 0.3,
    'colsample_bytree': 0.8
}


def grab_price_data(tickers_list):  
    full_price_history = []

    for ticker in tickers_list:
        stock_data = yf.download(ticker, period='2y', interval='1d')
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        stock_data['symbol'] = ticker
        full_price_history.append(stock_data)

    all_data = pd.concat(full_price_history)
    all_data.to_csv('price_data.csv', index=False)

def predict_next_day_prices(tickers):
    if not os.path.exists('price_data.csv'):
        grab_price_data(tickers)

    price_data = pd.read_csv('price_data.csv')

    # Assuming that the model requires certain features (adjust as per actual feature engineering steps)
    # Prepare the features and target variable
    price_data['return'] = price_data['close'].pct_change()
    price_data['target'] = (price_data['return'] > 0).astype(int)
    price_data.dropna(inplace=True)

    # Features and labels
    X = price_data[['open', 'high', 'low', 'close', 'volume']]
    y = price_data['target']

    # Split the data (assuming a basic split, adjust as per the original notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    model2 = RandomForestClassifier(n_estimators=100, oob_score=True, criterion="gini", random_state=0)
    model2.fit(X_train, y_train)

    # Make predictions for the next day
    next_day_prices = {}
    next_day_prices2 = {}

    for ticker in tickers:
        latest_data = yf.download(ticker, period='1d', interval='1d').iloc[-1]
        features = np.array([latest_data['Open'], latest_data['High'], latest_data['Low'], latest_data['Close'], latest_data['Volume']]).reshape(1, -1)
        prediction1 = model.predict(features)
        prediction2 = model2.predict(features)
        next_day_prices[ticker] = "Up" if prediction1[0] == 1 else "Down"
        next_day_prices2[ticker] = "Up" if prediction2[0] == 1 else "Down"
    
    shared_items = {k: next_day_prices[k] for k in next_day_prices if k in next_day_prices2 and next_day_prices[k] == next_day_prices2[k]}
    return shared_items


def append_data(json_file="predictions.json", shared_predictions=None):
    if shared_predictions is None:
        shared_predictions = {'AAPL': 'Up', 'GOOGL': 'Down'}  # Example default

    # Check if file exists and is not empty
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}  # Initialize with an empty dictionary if file is not found or empty

    # Update data with new predictions
    data[str(date.today())] = shared_predictions

    # Save the updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    

def fetch_closing_prices(symbols, prediction_date):
    # Ensure symbols is a list for proper handling by yfinance
    if isinstance(symbols, str):
        symbols = [symbols]

    # Check if the prediction_date is a Monday
    if prediction_date.weekday() == 0:  # 0 is Monday
        # Set start_date to the previous Friday
        start_date = prediction_date - timedelta(days=3)
    else:
        # Typically, the previous day's close is needed
        start_date = prediction_date - timedelta(days=1)

    end_date = prediction_date

    data = yf.download(symbols, start=start_date, end=end_date)
    # Ensure we return the Close prices for both days
    return data['Close']

def calculate_accuracy(json_file="predictions.json", prediction_date=date.today()):
    try:
        with open(json_file, 'r') as file:
            all_data = json.load(file)
            predictions = all_data.get(str(prediction_date))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON data: {e}")
        return

    if not predictions:
        print("No predictions found for the specified date.")
        return

    symbols = list(predictions.keys())
    closing_prices = fetch_closing_prices(symbols, prediction_date)

    # Check if closing_prices is a DataFrame or Series and handle accordingly
    if isinstance(closing_prices, pd.Series):
        # Convert Series to DataFrame if only one symbol's data was fetched
        closing_prices = closing_prices.to_frame().T

    correct_predictions = 0
    total_predictions = len(predictions)

    results = {}

    # Assuming the DataFrame has rows as dates and columns as symbols
    if len(closing_prices) > 1:
        previous_close = closing_prices.iloc[0]
        prediction_close = closing_prices.iloc[1]
    else:
        print("Insufficient data fetched for price comparison.")
        return

    for symbol, prediction in predictions.items():
        if symbol in prediction_close:
            is_correct = (prediction == 'Up' and prediction_close[symbol] > previous_close[symbol]) or \
                         (prediction == 'Down' and prediction_close[symbol] < previous_close[symbol])
            results[symbol] = "Correct" if is_correct else "Incorrect"
            if is_correct:
                correct_predictions += 1

    # Calculate and print overall accuracy
    accuracy_percentage = (correct_predictions / total_predictions) * 100
    print(f"Overall Accuracy: {accuracy_percentage:.2f}%")
    print("Individual Results:", results)
    

def run_script(tickers = ["AAPL", "JPM", "XLY"]):
    
    shared_predictions = predict_next_day_prices(tickers)
    
    print(len(shared_predictions))
    append_data(shared_predictions=shared_predictions)
    print(f'Predictions saved: {shared_predictions}')

    calculate_accuracy(prediction_date=date.today() - timedelta(days=1))
    
    return shared_predictions


# List of tickers to predict (personal testing)
tickers = ['AAPL', 'JPM', 'SPY', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'V',
           'JNJ', 'WMT', 'PG', 'UNH', 'NVDA', 'DIS', 'PYPL', 'HD', 'MA', 'VZ',
           'NFLX', 'INTC', 'MRK', 'PEP', 'KO', 'XOM', 'CSCO', 'ABT', 'CMCSA', 'ADBE',
           'CRM', 'PFE', 'AVGO', 'T', 'ABBV', 'NKE', 'COST', 'MDT', 'MCD', 'TXN',
           'HON', 'WFC', 'QCOM', 'BMY', 'UNP', 'PM', 'LLY', 'IBM', 'NEE', 'SBUX', 'XLY', "VYM"]

run_script(tickers)
