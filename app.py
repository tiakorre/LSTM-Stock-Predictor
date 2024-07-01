import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist.set_index('Date', inplace=True)
    hist.index = hist.index.tz_convert('UTC')  # Convert to the desired timezone
    return hist[['Close']]

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day, tzinfo=pytz.UTC)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    target_date = first_date

    dates, X, Y = [], [], []
    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return None
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        if next_week.empty:
            print('Error: No data for next week after', target_date)
            break
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year, month, day = map(int, next_date_str.split('-'))
        next_date = datetime.datetime(day=day, month=month, year=year, tzinfo=pytz.UTC)
        if last_time:
            break
        target_date = next_date
        if target_date == last_date:
            last_time = True
    ret_df = pd.DataFrame({'Target Date': dates})
    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]
    ret_df['Target'] = Y
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        print(f'Received ticker: {ticker}')
        data = get_stock_data(ticker)
        print('Data fetched successfully.')
        
        windowed_df = df_to_windowed_df(data, '2023-01-01', '2023-12-31', n=3)
        if windowed_df is None:
            return render_template('index.html', prediction='Error in processing data', image=None)
        
        print('Windowed dataframe created successfully.')
        dates, X, y = windowed_df_to_date_X_y(windowed_df)
        print('Dates, X, and y extracted successfully.')
        
        q_80 = int(len(dates) * .8)
        q_90 = int(len(dates) * .9)
        dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
        print('Training, validation, and test sets created successfully.')
        
        model = Sequential([
            LSTM(64, input_shape=(3, 1)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=1)
        print('Model trained successfully.')
        
        test_predictions = model.predict(X_test).flatten()
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test, test_predictions, label='Predictions')
        plt.plot(dates_test, y_test, label='Actual')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.legend()
        plt.savefig('static/prediction.png')
        plt.close()
        print('Prediction chart saved successfully.')
        
        next_day_prediction = model.predict(X_test[-1].reshape(1, 3, 1)).flatten()[0]
        return render_template('index.html', prediction=f'Next day prediction: {next_day_prediction}', image='static/prediction.png')
    return render_template('index.html', prediction=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)
