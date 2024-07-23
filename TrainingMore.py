import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
data = pd.read_csv("DataTrain.csv")

# Split features and labels
features = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll', 'Yaw']]
labels = data['Accident']

# Normalize features (example)
features = (features - features.mean()) / features.std()

# Convert to tensors
features_tensor = torch.tensor(features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

# Step 2: Define the Custom Dataset Class
class AccidentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset instance
dataset = AccidentDataset(features_tensor, labels_tensor)

# Split dataset into training and testing subsets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3: Create the PINN Model
class PINN(torch.nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = torch.nn.Linear(9, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.output_layer = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return torch.sigmoid(self.output_layer(x))
    
    def physics_informed_loss(self, x, mass):
        acc = x[:, 3:6]
        predicted_force = mass * acc
        expected_force = torch.zeros_like(predicted_force)
        physics_loss = torch.mean((predicted_force - expected_force) ** 2)
        return physics_loss

# Step 4: Train the Model
mass = 1.0
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

num_epochs = 100
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_features, batch_labels in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch_features)
        standard_loss = criterion(predictions, batch_labels)
        physics_loss = model.physics_informed_loss(batch_features, mass)
        total_loss = standard_loss + physics_loss
        total_loss.backward()
        optimizer.step()
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        for batch_features, batch_labels in test_dataloader:
            predictions = model(batch_features)
            standard_loss = criterion(predictions, batch_labels)
            physics_loss = model.physics_informed_loss(batch_features, mass)
            total_loss = standard_loss + physics_loss
            total_test_loss += total_loss.item()
            
            binary_predictions = (predictions > 0.5).float()
            correct_predictions += (binary_predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            all_predictions.append(predictions)
            all_labels.append(batch_labels)
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        accuracy = correct_predictions / total_samples
        print(f'Test Epoch [{epoch+1}/{num_epochs}], Loss: {avg_test_loss}, Accuracy: {accuracy*100:.2f}%')
        
# Concatenate all predictions and labels for visualization
all_predictions = torch.cat(all_predictions).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

# Convert probabilities to binary predictions
binary_predictions = (all_predictions > 0.5).astype(int)

# Sampling for better visualization (e.g., every 10th sample)
sample_indices = range(0, len(all_labels), 10)
sampled_labels = all_labels[sample_indices]
sampled_predictions = binary_predictions[sample_indices]

# Plot Predictions vs Actual
plt.figure(figsize=(14, 7))
plt.plot(sample_indices, sampled_labels, label='Actual', alpha=0.7, linestyle='-', marker='o')
plt.plot(sample_indices, sampled_predictions, label='Predicted', alpha=0.7, linestyle='-', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Accident (1) or No Accident (0)')
plt.title('Predictions vs Actual Values')
plt.show()
