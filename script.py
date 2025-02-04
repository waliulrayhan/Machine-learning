import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Function to load the Cora dataset
def load_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')  # Download and load the Cora dataset
    return dataset  # Return the dataset object

# Define the Graph Convolutional Network (GCN) Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation):
        """
        :param in_channels: Number of input features per node
        :param hidden_channels: Number of neurons in the hidden layer
        :param out_channels: Number of output classes (for classification)
        :param activation: Activation function (e.g., ReLU, Sigmoid, Tanh)
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # First Graph Convolution layer
        self.conv2 = GCNConv(hidden_channels, out_channels)  # Second Graph Convolution layer
        self.activation = activation  # Store the activation function

    def forward(self, data):
        """
        Forward pass of the GCN model.
        :param data: Graph data containing node features and edge information.
        """
        x, edge_index = data.x, data.edge_index  # Extract node features and edge index
        x = self.conv1(x, edge_index)  # Apply first GCN layer
        x = self.activation(x)  # Apply activation function (ReLU, Sigmoid, Tanh, etc.)
        x = self.conv2(x, edge_index)  # Apply second GCN layer
        return F.log_softmax(x, dim=1)  # Compute log-softmax for classification

# Function to select optimizer dynamically
def get_optimizer(optimizer_name, model, learning_rate, weight_decay):
    """
    :param optimizer_name: Optimizer type ('adam', 'sgd', 'rmsprop')
    :param model: The GNN model
    :param learning_rate: Learning rate for optimization
    :param weight_decay: L2 regularization term (helps prevent overfitting)
    :return: Selected optimizer instance
    """
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizers[optimizer_name](model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Function to select activation function dynamically
def get_activation_function(name):
    """
    :param name: Name of activation function ('relu', 'sigmoid', 'tanh')
    :return: Activation function
    """
    activations = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation function: {name}")
    return activations[name]

# Training function
def train(model, data, optimizer, criterion, epochs):
    """
    Trains the GNN model.
    :param model: GCN model
    :param data: Graph dataset
    :param optimizer: Optimizer (Adam, SGD, RMSProp)
    :param criterion: Loss function (CrossEntropyLoss)
    :param epochs: Number of training iterations
    """
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear previous gradients
        out = model(data)  # Forward pass
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute loss only for training nodes
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        if epoch % 100 == 0:  # Print loss every 100 epochs
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing function
def test(model, data):
    """
    Evaluates the trained model.
    :param model: Trained GNN model
    :param data: Graph dataset
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient calculation needed
        out = model(data)  # Forward pass
        pred = out.argmax(dim=1)  # Get predicted class labels
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # Count correct predictions
        accuracy = int(correct) / int(data.test_mask.sum())  # Compute accuracy

    # Display the results
    print("\n", "#" * 40)
    print("  Actual labels:   ", data.y[data.test_mask].tolist())
    print("  Predicted labels:", pred[data.test_mask].tolist())
    print("#" * 40)
    print(f" Accuracy: {accuracy:.4f}")
    print("#" * 40)

# Compute loss function
def calculate_loss(model, data, criterion):
    """
    Computes the loss on the test set.
    :param model: Trained model
    :param data: Graph dataset
    :param criterion: Loss function
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data.test_mask], data.y[data.test_mask])
        print(f"\nComputed Loss: {loss.item():.4f}")

# Main function to run the training and testing pipeline
def main(hyperparams):
    """
    Main function to execute training and evaluation.
    :param hyperparams: Dictionary containing model hyperparameters
    """
    dataset = load_data()  # Load the dataset
    data = dataset[0]  # Get graph data (single graph)

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the activation function
    activation_fn = get_activation_function(hyperparams["activation"])

    # Initialize the GCN model
    model = GCN(
        in_channels=dataset.num_node_features,  # Input feature dimension
        hidden_channels=hyperparams["hidden_channels"],  # Hidden layer size
        out_channels=dataset.num_classes,  # Number of output classes
        activation=activation_fn,  # Activation function
    ).to(device)

    # Move data to device (CPU/GPU)
    data = data.to(device)

    # Initialize optimizer
    optimizer = get_optimizer(hyperparams["optimizer"], model, hyperparams["learning_rate"], hyperparams["weight_decay"])

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train(model, data, optimizer, criterion, hyperparams["epochs"])

    # Test the trained model
    test(model, data)

    # Calculate loss after training
    calculate_loss(model, data, criterion)

# Hyperparameters configuration
if __name__ == "__main__":
    hyperparams = {
        "hidden_channels": 16,  # Number of hidden neurons
        "learning_rate": 0.01,  # Learning rate for optimization
        "weight_decay": 5e-4,  # L2 regularization
        "epochs": 1000,  # Number of training epochs
        "optimizer": "adam",  # Options: 'adam', 'sgd', 'rmsprop'
        "activation": "relu",  # Options: 'relu', 'sigmoid', 'tanh'
    }

    main(hyperparams)  # Run the model training and evaluation
