import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Data Preparation Function
def calculate_angles(data):
    pitch = np.arctan2(data['AccY'], np.sqrt(data['AccX']**2 + data['AccZ']**2)) * 180 / np.pi
    roll = np.arctan2(-data['AccX'], np.sqrt(data['AccY']**2 + data['AccZ']**2)) * 180 / np.pi
    return pitch, roll

# Load accelerometer and gyroscope data
acc_data = pd.read_csv('Train_DataAcc.csv')
gyro_data = pd.read_csv('Train_DataGyro.csv')
data = pd.concat([acc_data, gyro_data], axis=1)
delta_time = 0.02

# Calculate Pitch and Roll using accelerometer data
pitch, roll = calculate_angles(data)
data['Pitch'] = pitch
data['Roll'] = roll

# Calculate Yaw using GyroZ
yaw = np.zeros(len(gyro_data))
for i in range(1, len(gyro_data)):
    yaw[i] = yaw[i-1] + gyro_data['GyroZ'][i] * delta_time

# Add Yaw to accelerometer data
data['Yaw'] = yaw

# Prepare Input Features and Labels
input_features = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']].values
labels = data['Accident'].values.reshape(-1, 1)  # Assuming 'Accident' is the target column

# Convert to PyTorch tensors
inputs = torch.tensor(input_features, dtype=torch.float32)
targets = torch.tensor(labels, dtype=torch.float32)

# Define PINN model architecture
class PINN(nn.Module):
    def __init__(self, input_size):
        super(PINN, self).__init__()
        self.input_size = input_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid for binary classification (Accident or No Accident)
        )
    
    def forward(self, x):
        # Check input size
        assert x.shape[1] == self.input_size, f"Expected input size {self.input_size}, but got {x.shape[1]}"
        
        x = self.feature_extractor(x)
        
        # Check intermediate feature size
        assert x.shape[1] == 64, f"Expected intermediate feature size 64, but got {x.shape[1]}"
        
        x = self.predictor(x)
        
        # Check output size
        assert x.shape[1] == 1, f"Expected output size 1, but got {x.shape[1]}"
        
        return x

# Initialize PINN model
input_size = inputs.shape[1]
model = PINN(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = Adam(model.parameters(), lr=0.001)

# Prepare DataLoader
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs_batch, targets_batch in train_loader:
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
