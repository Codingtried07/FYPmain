import pandas as pd
df = pd.read_csv('C:/Users/renes/Downloads/mmWave-gesture-dataset-master/gesture_dataset/long_range_gesture/long_SEP/001/long_point_001_knock.csv', header=None)
print(df.shape)
print(df.head(20))