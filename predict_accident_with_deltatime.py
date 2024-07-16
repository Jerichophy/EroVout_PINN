import pandas as pd
import numpy as np
import torch
from torch import nn

# Function to generate test data
def generate_test_data(num_samples=500, delta_time=0.02):
    np.random.seed(42)
    test_data = {
        'Timestamp': np.arange(0, num_samples * delta_time, delta_time),
        'AccX': np.random.rand(num_samples),
        'AccY': np.random.rand(num_samples),
        'AccZ': np.random.rand(num_samples),
        'GyroX': np.random.rand(num_samples),
        'GyroY': np.random.rand(num_samples),
        'GyroZ': np.random.rand(num_samples),
        'Pitch': np.random.rand(num_samples) * 90,
        'Roll': np.random.rand(num_samples) * 90,
        'Yaw': np.cumsum(np.random.rand(num_samples) * 10)  # Cumulative Yaw over time
    }
    return pd.DataFrame(test_data)

# Load the trained model
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
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x

# Function to convert predictions to labels
def convert_to_label(predictions):
    return ["Accident" if pred >= 0.5 else "Safe" for pred in predictions]

# Main function for prediction testing
def main():
    # Generate test data
    test_df = generate_test_data(num_samples=500, delta_time=0.02)
    
    # Prepare input features for inference
    test_features = test_df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']].values
    inputs = torch.tensor(test_features, dtype=torch.float32)
    
    # Load the trained model and perform inference
    model = PINN(input_size=inputs.shape[1])
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.squeeze().numpy() 
    
    # Convert predictions to labels (Accident or Safe)
    labels = convert_to_label(predictions)
    
    # Add labels to the test_df DataFrame
    test_df['Prediction'] = labels
    
    # Print the test_df with Timestamp and Prediction columns
    print(test_df[['Timestamp', 'Prediction']])

if __name__ == "__main__":
    main()
