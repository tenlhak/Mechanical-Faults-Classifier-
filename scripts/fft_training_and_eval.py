

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from cnn1d_model import FaultClassifierCNN_frequency  # Ensure this import is correct

# Function to convert time-domain data to frequency domain
def time_to_frequency_domain(X):
    # Apply the real FFT along the time axis
    X_fft = np.fft.rfft(X, axis=2)
    # Compute the magnitude of the FFT coefficients
    X_magnitude = np.abs(X_fft)
    # Optionally, take the logarithm to compress the dynamic range
    X_log_magnitude = np.log1p(X_magnitude)
    return X_log_magnitude

# Load time-domain data
X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy')

# Convert to frequency domain
X_train_freq = time_to_frequency_domain(X_train)
X_test_freq = time_to_frequency_domain(X_test)

# Verify the shapes
print(f"X_train_freq shape: {X_train_freq.shape}")
print(f"X_test_freq shape: {X_test_freq.shape}")

# Normalize the data (recommended)
mean = X_train_freq.mean(axis=(0, 2), keepdims=True)
std = X_train_freq.std(axis=(0, 2), keepdims=True)
X_train_freq = (X_train_freq - mean) / std
X_test_freq = (X_test_freq - mean) / std  # Use training mean and std

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_freq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_freq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model and move it to the device
model = FaultClassifierCNN_frequency().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20  # Adjusted to 20 epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # Shape: (batch_size, channels, freq_bins)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average loss over the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")

# Evaluate on test data after training
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Optionally, save the trained model
torch.save(model.state_dict(), 'fault_classifier_fft.pth')
print("Model saved as 'fault_classifier_fft.pth'")
