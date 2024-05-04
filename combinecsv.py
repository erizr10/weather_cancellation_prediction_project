import pandas as pd

csv_files = ['Jan.csv', 'Feb.csv', 'Mar.csv', 'Apr.csv', 'May.csv', 'Jun.csv', 'Jul.csv', 'Aug.csv', 'Sep.csv', 'Oct.csv','Nov.csv','Dec.csv']

combined_df = pd.concat([pd.read_csv('../dataset/'+f) for f in csv_files])

filtered_df = combined_df[combined_df['ORIGIN'].isin(['JFK', 'LGA'])]

filtered_df.to_csv('filtered_combined.csv', index=False)
