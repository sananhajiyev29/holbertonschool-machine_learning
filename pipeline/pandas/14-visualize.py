#!/usr/bin/env python3
"""Visualize Bitcoin data with daily resampling from 2017 onward."""

import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file

# Load data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
if 'Weighted_Price' in df.columns:
    df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date and convert to datetime
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter from 2017 onward
df = df[df.index >= '2017-01-01']

# Resample to daily intervals
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
df_daily.plot(title='Bitcoin Daily Data from 2017')
plt.show()

# Return the processed daily DataFrame
df_daily
