import pandas as pd

flights_df = pd.read_csv('date_and_cancellation_rate.csv')
weather_df = pd.read_csv('3639124.csv')

flights_df['DATE'] = pd.to_datetime(flights_df['DATE'], format='%m/%d/%Y %I:%M:%S %p').dt.date
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%Y-%m-%d').dt.date


filtered_weather_df = weather_df[weather_df['NAME'] == 'JFK INTERNATIONAL AIRPORT, NY US']

combined_df = pd.merge(flights_df, filtered_weather_df, on='DATE', how='inner')
combined_df.to_csv('combined_data.csv', index=False)
