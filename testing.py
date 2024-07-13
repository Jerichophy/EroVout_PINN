import pandas as pd
import torch

Time = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataAcc.csv')['Time'].to_numpy())
dt = Time[1:] - Time[:-1]
print (dt)