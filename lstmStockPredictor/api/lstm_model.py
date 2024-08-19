import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

class LSTMStockPredictor:
    def __init__(self, company="AAPL", time_period="2y"):
        self.company = company
        self.time_period = time_period
        self.scaler_close = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler for Close price
        self.scaler_rsi = MinMaxScaler(feature_range=(0, 1))    # Initialize scaler for RSI
        self.scaler_indicator = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler for the new indicator
        self.model = None

    def compute_rsi(self, prices, period=20):
        delta = prices.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Step 1: Calculate the initial average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Initial RSI calculation
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Step 2: Calculate the RSI for subsequent periods using the smoothing factor
        for i in range(period, len(prices)):
            current_gain = gain.iloc[i]
            current_loss = loss.iloc[i]

            # Smoothing the average gain and loss
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + current_gain) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + current_loss) / period

            # RSI calculation using the smoothed average gain and loss
            rs = avg_gain.iloc[i] / avg_loss.iloc[i]
            rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi.fillna(50).values  # Fill any initial NaNs with a neutral value of 50

    def calculate_indicator(self, prices):
        indicator = []
        for n in range(1, len(prices)):
            # Calculate the percentage change from the previous day
            pc = (prices[n] - prices[n-1]) / prices[n-1]
            indicator.append(pc)
        
        # The first day has no previous day to compare, so set it to 0
        indicator.insert(0, 0)
        
        return np.array(indicator)

    
        
    

    def fetch_data(self):
        ticker = yf.Ticker(self.company)
        hist = ticker.history(period=self.time_period)
        hist.index = pd.to_datetime(hist.index)
        hist = hist.reset_index()
        hist.dropna(inplace=True)
        hist.reset_index(drop=True, inplace=True)
        
        # Calculate RSI
        hist['RSI'] = self.compute_rsi(hist['Close'])

        # Calculate the Indicator
        hist['Indicator'] = self.calculate_indicator(hist['Close'].values)
        
        self.hist_dates = hist['Date'].values  # Store the dates
        self.hist_close = hist[['Close']].values
        self.hist_rsi = hist[['RSI']].values
        self.hist_indicator = hist[['Indicator']].values

    def scale_data(self):
        # Scale Close prices, RSI, and the new indicator separately
        self.scaled_close = self.scaler_close.fit_transform(self.hist_close)
        self.scaled_rsi = self.scaler_rsi.fit_transform(self.hist_rsi)
        self.scaled_indicator = self.scaler_indicator.fit_transform(self.hist_indicator)
        
        # Combine scaled features
        self.scaled_data = np.hstack((self.scaled_close, self.scaled_rsi, self.scaled_indicator))
    
    def create_dataset(self, data, time_step=30):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step, :])  # Use all features
            y.append(data[i + time_step, 0])  # Target is still the Close price
        return np.array(X), np.array(y)
    
    def print_dataset(self, time_step=30):
        X, y = self.create_dataset(self.scaled_data, time_step)
        # Print a few samples from X and y
        print("X (features):")
        for i in range(min(5, len(X))):  # Print first 5 samples or less if not available
            print(X[i])
        print("\nY (targets):")
        for i in range(min(5, len(y))):  # Print first 5 targets or less if not available
            print(y[i])
        return X, y
    
    def train_model(self, time_step=30, epochs=20, batch_size=32, verbose=0):
        X, y = self.create_dataset(self.scaled_data, time_step)
        num_features = X.shape[2]
        X = X.reshape(X.shape[0], X.shape[1], num_features)  # Adjust input shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model = Sequential()
        self.model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], num_features)))
        self.model.add(LSTM(150))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        self.X_test = X_test
        self.y_test = y_test
        self.test_dates = self.hist_dates[-len(y_test):]  # Capture corresponding dates for y_test

    def predict_prices(self):
        predicted_prices_scaled = self.model.predict(self.X_test).flatten()
        return predicted_prices_scaled
    
    def inverse_transform(self, predicted_prices_scaled):
        # Ensure reshaping correctly
        predicted_prices_scaled = predicted_prices_scaled.reshape(-1, 1)
        # Combine with dummy RSI and Indicator (just for inverse transform consistency)
        combined = np.hstack((predicted_prices_scaled, np.zeros_like(predicted_prices_scaled), np.zeros_like(predicted_prices_scaled)))
        predicted_prices = self.scaler_close.inverse_transform(combined)[:, 0]
        return predicted_prices

    def calculate_error(self, predicted_price, actual_price):
        error = abs(predicted_price - actual_price)
        error_percentage = (error / actual_price) * 100
        return error_percentage
    
    def get_results(self):
        predicted_prices_scaled = self.predict_prices()
        predicted_prices = self.inverse_transform(predicted_prices_scaled)
        
        # Ensure y_test is properly reshaped for comparison
        y_test_original = self.scaler_close.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        if predicted_prices.shape != y_test_original.shape:
            print(f"Shape mismatch: predicted {predicted_prices.shape}, actual {y_test_original.shape}")
            return None
    
        error_percentage = self.calculate_error(predicted_prices, y_test_original)
        
        stock_results = pd.DataFrame({
            'Date': self.test_dates[-len(predicted_prices):],  # Match the dates to predicted prices
            'Actual': y_test_original,
            
            'Predicted': predicted_prices,
            'Error Percentage': error_percentage
        })
        
        return stock_results
    def calculate_rmse(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse

    def plot_rmse(self,actual, predicted):
        # Inverse transform predictions and actual values to get them back to their original scale
       
        
        rmse_values = [self.calculate_rmse(actual[:i+1], predicted[:i+1]) for i in range(len(actual))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rmse_values, label='RMSE Over Time')
        plt.xlabel('Days')
        plt.ylabel('RMSE')
        plt.title('Root Mean Squared Error Over Time')
        plt.legend()
        plt.show()

# Example usage:
predictor = LSTMStockPredictor(company="AAPL")
predictor.fetch_data()
predictor.scale_data()

# Verify the data shape and contents
# X, y = predictor.print_dataset()  # Print the dataset to verify

# Train the model and get results
predictor.train_model()
results = predictor.get_results()

# Print the final results
# print("\nFinal Results:")
# print(results)
if results is not None:
    actual_prices = results['Actual'].values
    predicted_prices = results['Predicted'].values

    # Calculate and plot RMSE
    predictor.plot_rmse(actual_prices, predicted_prices)