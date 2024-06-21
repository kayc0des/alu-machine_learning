#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

# Rename a column using .rename
df = df.rename(columns={'Timestamp': 'Datetime'})

# Convert Timestamp values to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

df = df[['Datetime', 'Close']]

print(df)
