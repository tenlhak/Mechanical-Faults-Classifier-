import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from cnn1d_model import FaultClassifierCNN_time # Ensure this import is correct


import numpy as np
import torch



X_train = np.load('X_train.npy', allow_pickle=True)



y_train = np.load('y_train.npy')

X_test = np.load('X_test.npy', allow_pickle=True)



y_test = np.load('y_test.npy')

# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.load('y_train.npy'), dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(np.load('y_test.npy'), dtype=torch.long)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32  # Adjust based on your system's memory capacity

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''Traning the model on the data'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FaultClassifierCNN_time().to(device)         #accuracy 0.85


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20  # Adjust based on convergence

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions.double() / len(train_dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')



'''Evalutaing the model'''

model.eval()
correct_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #print(f'\nThe prediction is {preds} and the label is {labels}')
        correct_predictions += torch.sum(preds == labels)

test_acc = correct_predictions.double() / len(test_dataset)
print(f'Test Accuracy: {test_acc:.4f}')
