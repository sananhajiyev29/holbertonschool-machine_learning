# Time Series Forecasting

This project forecasts the closing price of Bitcoin (BTC) using a
recurrent neural network. The model uses the past 24 hours of BTC data
to predict the closing price of the following hour.

## Files

- `preprocess_data.py`: cleans and preprocesses the raw Coinbase/Bitstamp
  datasets. It drops missing rows, keeps the informative price and volume
  features, downsamples the 1-minute data to 1-hour windows, normalizes
  the data using training statistics, and builds 24-hour input sequences
  paired with the next hour's closing price. The result is saved to a
  compressed `.npz` file.
- `forecast_btc.py`: loads the preprocessed data, feeds it to the model
  through a `tf.data.Dataset`, builds an LSTM-based RNN, trains it with
  mean-squared error as the cost function, validates it, and saves the
  trained model.

## Preprocessing decisions

- **Not all data points are useful**: rows with missing values are
  dropped, and the 1-minute resolution is downsampled to 1-hour windows
  since the goal is hourly forecasting.
- **Not all features are useful**: only the close price, BTC volume,
  currency volume, and weighted price are retained.
- **Rescaling**: features are standardized (zero mean, unit variance)
  using statistics computed on the training set only, to avoid leaking
  validation information.
- **Window relevance**: each sample uses a sliding window of the past 24
  hours to predict the next hour's close.

## Usage
./preprocess_data.py

./forecast_btc.py
