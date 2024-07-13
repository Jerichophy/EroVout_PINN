import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Normalize data
def normalize_data(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    normalized_data = (data - mean) / std
    return normalized_data

# Define the equations for the angles from the accelerometer data
def calculate_angles(accX, accY, accZ):
    AngX = torch.atan(accY / torch.sqrt(accX**2 + accZ**2)) * 180 / np.pi
    AngY = torch.atan(-accX / torch.sqrt(accY**2 + accZ**2)) * 180 / np.pi
    return AngX, AngY

# Define the fully connected network
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

torch.manual_seed(123)

# DATASET
accX = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataAcc.csv')['AccX'].to_numpy())
accY = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataAcc.csv')['AccY'].to_numpy())
accZ = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataAcc.csv')['AccZ'].to_numpy())
gyroX = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataGyro.csv')['GyroX'].to_numpy())
gyroY = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataGyro.csv')['GyroY'].to_numpy())
gyroZ = torch.from_numpy(pd.read_csv(r'C:\Users\Fushio\Desktop\Untitled Folder\EroVout_PINN\Train_DataGyro.csv')['GyroZ'].to_numpy())
dt = torch.tensor([0.02])

# Define the neural network
N_INPUT = 9
N_OUTPUT = 1
N_HIDDEN = 50
N_LAYERS = 3
pinn = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS)

# Calculate angles
Pitch, Roll = calculate_angles(accX, accY, accZ)
Yaw = torch.cumsum(gyroZ * dt, dim=0)

# Stack the inputs (e.g., the sensor readings)
inputs = torch.stack((accX, accY, accZ, gyroX, gyroY, gyroZ, Pitch, Roll, Yaw), dim=-1)

# Normalize inputs
inputs = normalize_data(inputs.view(-1, N_INPUT))  # Reshape inputs to match N_INPUT

# Generate synthetic labels based on thresholds
threshold_angle = 30  # degrees
threshold_yaw_rate = 5  # degrees/second
labels = ((Pitch.abs() > threshold_angle) | (Roll.abs() > threshold_angle) | (Yaw.abs() > threshold_yaw_rate)).float().view(-1, 1)

# Convert data to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Early stopping parameters
patience = 10
best_loss = float('inf')

# Train the PINN
num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    predictions = pinn(inputs)
    loss = criterion(predictions, labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update best loss and stop if no improvement after patience epochs
    if loss.item() < best_loss:
        best_loss = loss.item()
    else:
        if epoch >= patience:
            break

    # Print loss and evaluation metrics every 5000 epochs
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        with torch.no_grad():
            pred_labels = torch.sigmoid(predictions) > 0.5
            accuracy = accuracy_score(labels, pred_labels.numpy())
            precision = precision_score(labels, pred_labels.numpy())
            recall = recall_score(labels, pred_labels.numpy())
            f1 = f1_score(labels, pred_labels.numpy())
            print(f'Accuracy: {accuracy * 100:.2f}%')
            print(f'Precision: {precision:.2f}')
            print(f'Recall: {recall:.2f}')
            print(f'F1-score: {f1:.2f}')
            print()

# Output example of predictions
print("Example predictions (1 for accident-prone, 0 for normal):")
print((torch.sigmoid(pinn(inputs)) > 0.5).view(-1))

# Function to test new data
def test_new_data(new_data):
    new_data = torch.tensor(new_data, dtype=torch.float32).view(1, -1)
    accX, accY, accZ, gyroX, gyroY, gyroZ, time = new_data[:, :1], new_data[:, 1:2], new_data[:, 2:3], new_data[:, 3:4], new_data[:, 4:5], new_data[:, 5:6], new_data[:, 6:7]
    
    # Calculate Pitch, Roll, and Yaw for new data
    Pitch, Roll = calculate_angles(accX, accY, accZ)
    dt = time[0, 0] - (time[0, 0] - 0.01)
    Yaw = gyroZ * dt

    # Stack inputs for new data
    new_inputs = torch.hstack((accX, accY, accZ, gyroX, gyroY, gyroZ, Pitch, Roll, Yaw))

    with torch.no_grad():
        prediction = torch.sigmoid(pinn(new_inputs))
        is_accident_prone = prediction > 0.5
        return is_accident_prone.item()

# Example of testing new data
new_data_example = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1]
is_accident_prone = test_new_data(new_data_example)
print(f"New data is {'accident-prone' if is_accident_prone else 'not accident-prone'}")
