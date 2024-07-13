import numpy as np
import pandas as pd

GyroFile_Path = 'archive/First route/First lap/BS_Route1_gyroscope_1.csv'
AccFile_Path = 'archive/First route/First lap/BS_Route1_accelerometer_1.csv'


def load_data(GyroFile_Path):
    # Gyro Stuffs 
    dfGyro = pd.read_csv(GyroFile_Path, header=None, sep=';', nrows = 24793)
    # Convert DataFrame to a NumPy array
    new_dfGyro = dfGyro.iloc[:, 2:5]
    new_dfGyro.columns = ['GyroX', 'GyroY', 'GyroZ']
    new_file_pathGyro = 'Train_DataGyro.csv'
    new_dfGyro.to_csv(new_file_pathGyro, index=False)

    # Accelerometer Stuff with time
    dfAcc = pd.read_csv(AccFile_Path, header=None, sep=';', nrows = 24793)
    # Convert DataFrame to a NumPy array
    new_dfAcc = dfAcc.iloc[:, [2,4,5]]
    new_dfAcc.columns = ['AccX', 'AccY', 'AccZ']
    new_file_pathAcc = 'Train_DataAcc.csv'
    new_dfAcc.to_csv(new_file_pathAcc, index=False)
    
    return new_file_pathGyro, new_file_pathAcc

print (load_data(GyroFile_Path))