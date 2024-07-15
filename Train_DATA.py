import numpy as np
import pandas as pd
import random

# Gyro
GyroFile_Path = 'archive/First route/First lap/BS_Route1_gyroscope_1.csv'

# Acc
AccFile_Path = 'archive/First route/First lap/BS_Route1_accelerometer_1.csv'

def load_data(GyroFile_Path, AccFile_Path):
    # Load Gyroscope data
    dfGyro = pd.read_csv(GyroFile_Path, header=None, sep=';', nrows=24793)
    new_dfGyro = dfGyro.iloc[:, 2:5]
    new_dfGyro.columns = ['GyroX', 'GyroY', 'GyroZ']
    new_file_pathGyro = 'Train_DataGyro.csv'
    new_dfGyro.to_csv(new_file_pathGyro, index=False)

    # Load Accelerometer data
    dfAcc = pd.read_csv(AccFile_Path, header=None, sep=';', nrows=24793)
    new_dfAcc = dfAcc.iloc[:, [2, 4, 5]]
    new_dfAcc.columns = ['AccX', 'AccY', 'AccZ']
    new_dfAcc['Accident'] = np.random.randint(2, size=len(new_dfAcc)) 
    new_file_pathAcc = 'Train_DataAcc.csv'
    new_dfAcc.to_csv(new_file_pathAcc, index=False)
    
    return new_file_pathGyro, new_file_pathAcc
