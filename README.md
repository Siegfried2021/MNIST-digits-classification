# MNIST Digit Classification with PyTorch

This project implements a neural network in PyTorch for classifying handwritten digits from the MNIST dataset. Using a fully connected network with two hidden layers, this model is trained to recognize and classify digits from 0 to 9, achieving high accuracy on validation data.

## Overview

The model uses the classic MNIST dataset, which contains images of digits (28x28 pixels, grayscale) and their respective labels (0–9). This project includes code to load and preprocess the dataset, build and train the model, evaluate its performance, and save the trained model for future use.

## How the Model Works

The model is a simple neural network classifier designed as follows:
- **Input Layer**: Accepts each 28x28 image, flattened into a 784-dimensional vector.
- **Hidden Layers**: Two fully connected layers with 1024 and 512 units, respectively, both followed by ReLU activation for non-linearity.
- **Output Layer**: A 10-unit layer with LogSoftmax activation, providing the probability distribution over the 10 possible digit classes.

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
