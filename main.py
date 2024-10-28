import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import struct
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Set up the dataset path
data = '/home/mathieulecouvet/Desktop/BeCode_AI/Projets/MNIST-digits-classification/data'

# Check if the dataset folder and files exist
print("Files in dataset folder:", os.listdir(data))

# Define paths for each MNIST file
train_images_path = os.path.join(data, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data, 't10k-labels.idx1-ubyte')

def read_idx(filename):
    """
    Reads an IDX file and returns the data as a NumPy array.

    Parameters:
    - filename (str): The path to the IDX file.

    Returns:
    - np.array: The data from the IDX file as a NumPy array.
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_data():
    """
    Loads and preprocesses the MNIST dataset.

    Returns:
    - tuple: DataLoaders for training and validation data.
    """
    # Load dataset from IDX files
    X_train = read_idx(train_images_path)
    y_train = read_idx(train_labels_path)
    X_val = read_idx(test_images_path)
    y_val = read_idx(test_labels_path)
    
    # Normalize and convert data to tensors
    X_train, X_val = torch.tensor(X_train / 255.0, dtype=torch.float32), torch.tensor(X_val / 255.0, dtype=torch.float32)
    y_train, y_val = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)
    
    # Add channel dimension for PyTorch (1, 28, 28)
    X_train, X_val = X_train.unsqueeze(1), X_val.unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader

class MNISTModel(nn.Module):
    """
    Neural network model for digit classification using PyTorch.
    """
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax for use with NLLLoss

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def train_model(model, train_loader, val_loader, device, epochs=20):
    """
    Trains the model on the provided training data.

    Parameters:
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - device (torch.device): Device to train on (CPU or GPU).
    - epochs (int): Number of training epochs.
    """
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Final validation accuracy after training
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Final Validation Accuracy: {accuracy:.2f}%')

def main():
    """
    Main function to orchestrate the loading, training, and saving of the MNIST model.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader = load_data()
    
    # Initialize model
    model = MNISTModel()
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model training complete and saved as 'mnist_model.pth'.")

# Run the main function if this script is executed
if __name__ == '__main__':
    main()
