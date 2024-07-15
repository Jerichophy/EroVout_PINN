import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Load data
acc_data = pd.read_csv('Train_DataAcc.csv')
gyro_data = pd.read_csv('Train_DataGyro.csv')

# Combine data
data = pd.concat([acc_data, gyro_data], axis=1)
print(data)