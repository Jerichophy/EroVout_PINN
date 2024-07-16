import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from scipy.integrate import odeint

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

# Add Yaw to accelerometer data
data['Yaw'] = yaw

# Prepare Input Features and Labels
input_features = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']].values
labels = data['Accident'].values.reshape(-1, 1)  # Assuming 'Accident' is the target column

# Convert to PyTorch tensors
inputs = torch.tensor(input_features, dtype=torch.float32)
targets = torch.tensor(labels, dtype=torch.float32)

# Define Enhanced PINN model architecture
class EnhancedPinn(nn.Module):
    def __init__(self, input_size):
        super(EnhancedPinn, self).__init__()
        self.input_size = input_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),  # Increased capacity
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid for binary classification (Accident or No Accident)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x

# Initialize Enhanced PINN model
input_size = inputs.shape[1]
model = EnhancedPinn(input_size)

# Define loss function and optimizer
learning_rate = 0.0001
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = Adam(model.parameters(), lr=learning_rate)

# Prepare DataLoader
BatchSize = 64
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True)

# Define the enhanced physics-informed loss function
def enhanced_physics_loss(acc, vel, predicted_force, mass=1.0, damping=0.1):
    # Simulate velocity change over time
    vel_next = vel + predicted_force / mass * delta_time - damping * vel * delta_time
    
    # Calculate actual acceleration from velocity changes
    actual_acceleration = (vel_next - vel) / delta_time
    
    # Loss based on the difference between predicted and actual accelerations
    loss = torch.mean((actual_acceleration - acc)**2)
    return loss

def returns_velocity_dynamics(y, t, predicted_force, mass=1.0, damping=0.1):
    dydt = predicted_force / mass - damping * y #Damping + Newtons 2nd law
    return dydt.squeeze()  

# Training Loop with enhanced physics-informed loss using SciPy's odeint
num_epochs = 10
mass = 1.0 
damping = 0.1  

for epoch in range(num_epochs):
    for inputs_batch, targets_batch in train_loader:
        outputs = model(inputs_batch)
        
        # Calculate the standard loss
        loss = criterion(outputs, targets_batch)
        
        # Extract acceleration for the batch
        acc = inputs_batch[:, :3]  # AccX, AccY, AccZ
        
        # Detach the tensor from the computation graph and convert to NumPy array
        predicted_force_np = outputs.cpu().detach().numpy()
        
        # Run the ODE solver to simulate velocity dynamics
        t = np.linspace(0, delta_time, num=len(acc))  # Time steps
        velocities = odeint(returns_velocity_dynamics, np.zeros_like(acc[:, 0]), t, args=(predicted_force_np, mass, damping))
        
        # Incorporate enhanced physics-informed loss using simulated velocities
        phy_loss = enhanced_physics_loss(acc, velocities[-1], predicted_force_np, mass, damping)
        
        # Total loss
        total_loss = loss + phy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'sciipy_odeint_trained_model.pth')
