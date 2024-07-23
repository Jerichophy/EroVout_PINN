import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('DataTrain.csv')

# Feature columns and target column
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']
target = 'Accident'

# Split into features and target
X = df[features].values
y = df[target].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors and enable autograd
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

model = PINN()

# Define the revised Newton's second law loss function
def newton_loss(model, X, y, mass=1.0):
    # Predictions
    y_pred = model(X)
    
    # Calculate the magnitude of acceleration from the accelerometer data
    acc_x, acc_y, acc_z = X[:, 2], X[:, 3], X[:, 4]
    acc_magnitude = torch.sqrt(acc_x**2 + acc_y**2 + acc_z**2).view(-1, 1)
    
    # Calculate the predicted force from the model
    force_pred = y_pred
    
    # Calculate the force using Newton's second law: F = ma
    force_true = mass * acc_magnitude
    
    # Penalize the difference between the predicted force and the force obtained from Newton's second law
    newton_loss = torch.mean((force_pred - force_true) ** 2)
    
    # Binary Cross-Entropy Loss for the prediction
    bce_loss = nn.BCELoss()(y_pred, y)
    
    # Total loss
    total_loss = bce_loss + newton_loss
    return total_loss

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with Newton's second law loss
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    loss = newton_loss(model, X_train_tensor, y_train_tensor, mass=1.0)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Model evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Convert predictions to binary outcomes
predictions = (predictions > 0.5).astype(int)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], 'b', label='Actual')
plt.plot(predictions[:100], 'r', label='Predicted')
plt.xlabel('Sample index')
plt.ylabel('Accident')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'pinn_model.pth')
