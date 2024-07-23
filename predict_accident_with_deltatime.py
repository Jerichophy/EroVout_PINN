import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the model class
class PINNStuff(nn.Module):
    def __init__(self, input_dim):
        super(PINNStuff, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# Load the trained model
input_dim = 9  # Number of input features
model = PINNStuff(input_dim)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Function to predict accident
def predict_accident(data):
    # Features
    X = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred_class = (y_pred > 0.5).float()
    
    return y_pred_class.numpy(), y_pred.numpy()

# Example usage with random data
if __name__ == "__main__":
    # Generate random data for testing
    np.random.seed(42)  # For reproducibility
    random_data = {
        'AccX': np.random.randn(10),
        'AccY': np.random.randn(10),
        'AccZ': np.random.randn(10),
        'GyroX': np.random.randn(10),
        'GyroY': np.random.randn(10),
        'GyroZ': np.random.randn(10),
        'Pitch': np.random.randn(10),
        'Roll': np.random.randn(10),
        'Yaw': np.random.randn(10)
    }
    new_data = pd.DataFrame(random_data)

    # Predict accidents
    predictions_class, predictions_prob = predict_accident(new_data)

    # Print the predictions
    print("Predictions:")
    for i, pred in enumerate(predictions_class):
        if pred == 1:
            print(f"Sample {i+1}: There will be an accident")
        else:
            print(f"Sample {i+1}: No accident in the near future")

    # Plot the predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_prob, label='Predicted Probability', marker='o')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    for i, prob in enumerate(predictions_prob):
        plt.text(i, prob, f'{prob[0]:.2f}', ha='center', va='bottom' if prob > 0.5 else 'top', color='blue')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Accident Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
