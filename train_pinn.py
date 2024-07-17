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
new_file_data = 'CombinedData.csv'
data.to_csv(new_file_data, index=False)
delta_time = 0.02

# Calculate Pitch and Roll using accelerometer data
pitch, roll = calculate_angles(data)
data['Pitch'] = pitch
data['Roll'] = roll

# Calculate Yaw using GyroZ
yaw = np.zeros(len(gyro_data))
for i in range(1, len(gyro_data)):
    yaw[i] = yaw[i-1] + gyro_data['GyroZ'][i] * delta_time  # Yaw += GyroZ*DeltaTime

data['Yaw'] = yaw

# Calculate Velocity from Acceleration
velocity = np.zeros((len(acc_data), 3))  # Initialize velocity array for [VelX, VelY, VelZ]
for i in range(1, len(acc_data)):
    velocity[i] = velocity[i-1] + acc_data[['AccX', 'AccY', 'AccZ']].iloc[i] * delta_time  # Velocity += Acceleration*DeltaTime

data['VelX'] = velocity[:, 0]
data['VelY'] = velocity[:, 1]
data['VelZ'] = velocity[:, 2]

input_features = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw', 'VelX', 'VelY', 'VelZ']].values
labels = data['Accident'].values.reshape(-1, 1) 

# Convert to PyTorch tensors
inputs = torch.tensor(input_features, dtype=torch.float32)
targets = torch.tensor(labels, dtype=torch.float32)

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
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x

# Initialize PINN model
input_size = inputs.shape[1]
model = PINN(input_size)

# Define loss function and optimizer
learning_rate = 0.0001
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = Adam(model.parameters(), lr=learning_rate)

# Prepare DataLoader
BatchSize = 64
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True)

# Define the physical loss functions
def physics_loss(acc, predicted_force, mass=1.0):
    # Newton's second law: F = ma
    force = mass * acc
    return torch.mean((force - predicted_force)**2)

def dynamic_loss(vel, acc, delta_time):
    # First-order dynamic model: Vel += Acc * DeltaTime
    predicted_vel = vel[:, :-1] + acc[:, 1:] * delta_time
    return torch.mean((predicted_vel - vel[:, 1:])**2)

# Training Loop with physics-informed loss
num_epochs = 10
mass = 1.0  

for epoch in range(num_epochs):
    for inputs_batch, targets_batch in train_loader:
        outputs = model(inputs_batch)
        
        # Calculate the standard loss
        loss = criterion(outputs, targets_batch)
        
        # Incorporate physics-informed loss
        acc = inputs_batch[:, :3]  # AccX, AccY, AccZ
        vel = inputs_batch[:, -3:]  # VelX, VelY, VelZ
        
        # Convert predicted force to acceleration to match dimensions
        predicted_acc = outputs * mass 
        phy_loss = physics_loss(acc, predicted_acc, mass)
        
        # Calculate dynamic loss
        dyn_loss = dynamic_loss(vel, acc, delta_time)
        
        # Total loss
        total_loss = loss + phy_loss + dyn_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
