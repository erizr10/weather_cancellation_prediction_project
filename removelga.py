import pandas as pd

df = pd.read_csv('combined_data.csv')

filtered_df = df[df['STATION'] != 'USW00014732']
filtered_df.to_csv('combined_data.csv', index=False)