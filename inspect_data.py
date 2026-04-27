import pandas as pd
df = pd.read_csv('data/001/long_point_001_knock.csv', header=None)
print(df.shape)
print(df.head(20))