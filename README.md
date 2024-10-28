# MNIST Digit Classification with PyTorch

This project implements a neural network in PyTorch for classifying handwritten digits from the MNIST dataset. Using a fully connected network with two hidden layers, this model is trained to recognize and classify digits from 0 to 9, achieving high accuracy on validation data.

## Overview

The model uses the classic MNIST dataset, which contains images of digits (28x28 pixels, grayscale) and their respective labels (0–9). This project includes code to load and preprocess the dataset, build and train the model, evaluate its performance, and save the trained model for future use.

## How the Model Works

The model is a fully connected neural network built using PyTorch, specifically designed to classify the MNIST dataset of handwritten digits. The architecture includes two hidden layers with ReLU activations and an output layer with LogSoftmax activation for multi-class classification.

### Architecture Details

- **Input Layer**: 
  - Accepts a 28x28 grayscale image, flattened into a 784-dimensional vector.

- **Hidden Layer 1**:
  - **Units**: 1024
  - **Activation**: ReLU
  - **Purpose**: Learns complex patterns by introducing non-linearity. The high number of units allows the model to capture various details and features of the digits.

- **Hidden Layer 2**:
  - **Units**: 512
  - **Activation**: ReLU
  - **Purpose**: Further refines patterns learned in the first layer, focusing on higher-level features that help distinguish digit classes.

- **Output Layer**:
  - **Units**: 10 (one for each digit class, 0–9)
  - **Activation**: LogSoftmax
  - **Purpose**: Outputs log-probabilities for each class, suitable for multi-class classification tasks with `NLLLoss` (Negative Log Likelihood Loss).

### Summary Table

| Layer              | Units | Activation | Output Shape |
|--------------------|-------|------------|--------------|
| Input Layer        | 784   | -          | (batch_size, 784) |
| Hidden Layer 1     | 1024  | ReLU       | (batch_size, 1024) |
| Hidden Layer 2     | 512   | ReLU       | (batch_size, 512) |
| Output Layer       | 10    | LogSoftmax | (batch_size, 10) |

### Loss and Optimization

- **Loss Function**: Negative Log Likelihood Loss (NLLLoss), which works with LogSoftmax to compute the probability of the correct class and penalize incorrect predictions.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001, chosen for its efficiency in handling sparse gradients and adaptability to a variety of datasets.

### Model Workflow

1. **Flattening**: Each 28x28 image is flattened to a 784-dimensional vector.
2. **Feedforward**: The data passes through two hidden layers with ReLU activation to learn non-linear features.
3. **Classification**: The output layer applies LogSoftmax, converting the output to log-probabilities for each class, which are then used to predict the most likely digit.

This architecture balances simplicity and power, enabling high accuracy on the MNIST dataset while remaining computationally efficient.

**Objective**: The network minimizes the negative log likelihood loss (NLLLoss), aiming to maximize the probability of the correct class.

## Project Structure

- **Data Loading**: Functions are provided to read and preprocess data from the MNIST dataset files in IDX format.
- **Model Architecture**: A fully connected neural network with two hidden layers.
- **Training & Validation**: The model is trained on the training set and validated on the test set for accuracy.
- **Model Saving**: After training, the model is saved for later use in inference tasks.

## Code Overview

### Main Components

- **Data Loading**:
  - `read_idx`: Reads IDX files and returns the data as a NumPy array.
  - `load_data`: Loads and normalizes images, adds necessary channel dimensions, and creates `DataLoader` instances for efficient batch processing.

- **Model Architecture**:
  - `MNISTModel`: Defines a simple neural network with two hidden layers, ReLU activations, and a final LogSoftmax layer.

- **Training**:
  - `train_model`: Implements the training loop, using the Adam optimizer and NLLLoss. After training, it evaluates the model on the validation set and prints the final accuracy.

- **Saving the Model**:
  - After training, the model’s state dictionary is saved as `mnist_model.pth` for reuse in prediction or inference tasks.

## Running the Code

To execute the code, run the `main.py` file:

```bash
python main.py
```

## Expected output

```plaintext
Files in dataset folder: ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']
Epoch 1: ...
Final Validation Accuracy: 98.30%
Model training complete and saved as 'mnist_model.pth'.
