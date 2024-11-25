import numpy as np

class NeuronNetwork:
    def __init__(self):  # Corrected constructor name with double underscores
        # Initialize weights and inputs as given in the problem
        self.weights = np.array([0.1, 0.3, -0.2])  # w1, w2, w3
        self.inputs = np.array([0.8, 0.6, 0.4])    # x1, x2, x3
        self.bias = 0.35                           # b

    def binary_sigmoid(self, x):
        """Implements the binary sigmoid activation function f(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-x))

    def bipolar_sigmoid(self, x):
        """Implements the bipolar sigmoid activation function f(x) = (1 - e^(-x)) / (1 + e^(-x))"""
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def identity(self, x):
        """Implements the identity activation function f(x) = x"""
        return x

    def threshold(self, x):
        """Implements the threshold activation function f(x) = 1 if x >= 0 else 0"""
        return 1 if x >= 0 else 0

    def relu(self, x):
        """Implements the ReLU activation function f(x) = max(0, x)"""
        return max(0, x)

    def hyperbolic_tangent(self, x):
        """Implements the hyperbolic tangent activation function f(x) = tanh(x)"""
        return np.tanh(x)

    def calculate_net_input(self):
        """Calculates the net input to the neuron y_in = b + Σ(xi * wi)"""
        return self.bias + np.sum(self.inputs * self.weights)

    def compute_outputs(self):
        """Computes all the activation function outputs"""
        # Calculate the net input (y_in)
        net_input = self.calculate_net_input()

        # Calculate outputs using all activation functions
        binary_output = self.binary_sigmoid(net_input)
        bipolar_output = self.bipolar_sigmoid(net_input)
        identity_output = self.identity(net_input)
        threshold_output = self.threshold(net_input)
        relu_output = self.relu(net_input)
        hyperbolic_tangent_output = self.hyperbolic_tangent(net_input)

        return {
            'net_input': net_input,
            'binary_sigmoid': binary_output,
            'bipolar_sigmoid': bipolar_output,
            'identity': identity_output,
            'threshold': threshold_output,
            'relu': relu_output,
            'hyperbolic_tangent': hyperbolic_tangent_output
        }

# Create and use the network
network = NeuronNetwork()
results = network.compute_outputs()

# Print results
print(f"Net input (y_in): {results['net_input']:.3f}")
print(f"Binary sigmoid output: {results['binary_sigmoid']:.3f}")
print(f"Bipolar sigmoid output: {results['bipolar_sigmoid']:.3f}")
print(f"Identity output: {results['identity']:.3f}")
print(f"Threshold output: {results['threshold']}")
print(f"ReLU output: {results['relu']:.3f}")
print(f"Hyperbolic tangent output: {results['hyperbolic_tangent']:.3f}")