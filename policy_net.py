import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


def hidden_init(layer):
    """Initialize weights for hidden layers with uniform distribution"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Create a multi-layer perceptron (MLP)

    Args:
        sizes (list): List of layer sizes [input_dim, hidden1_dim, ..., output_dim]
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer (default: Identity)

    Returns:
        nn.Sequential: Constructed MLP network
    """
    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
    # Add output layer
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    net = nn.Sequential(*layers)

    # Initialize weights
    for i in range(len(net)):
        if isinstance(net[i], nn.Linear):
            # Use Kaiming initialization for ReLU/LeakyReLU
            if i + 1 < len(net) and isinstance(net[i + 1], nn.ReLU):
                nn.init.kaiming_normal_(net[i].weight, nonlinearity='relu')
            elif i + 1 < len(net) and isinstance(net[i + 1], nn.LeakyReLU):
                nn.init.kaiming_normal_(net[i].weight, nonlinearity='leaky_relu')
            # Use Xavier initialization for other activations
            else:
                nn.init.xavier_normal_(net[i].weight)
            # Initialize biases to zero
            nn.init.zeros_(net[i].bias)
    return net


def backward_hook(module, gout):
    """
    Backward hook to add regularization loss during backpropagation

    Args:
        module: The module to apply the hook
        gout: Gradients from the next layer

    Returns:
        Modified gradients
    """
    if hasattr(module, 'regularization_loss') and module.regularization_loss != 0:
        module.regularization_loss.backward(retain_graph=True)
        module.regularization_loss = 0
    return gout


class Actor(nn.Module):
    """
    Actor network for reinforcement learning with Fourier feature extraction

    Args:
        state_size (int): Dimensionality of state space
        action_size (int): Dimensionality of action space
        hidden_size (int): Number of units in hidden layers (default: 32)
        ptff_params (dict): Parameters for Fourier feature extractor (default: None)
    """

    def __init__(self, state_size, action_size, hidden_size=32, ptff_params=None):
        super(Actor, self).__init__()

        # Default parameters for Fourier feature extractor
        if ptff_params is None:
            ptff_params = {
                'sizes': [state_size, hidden_size, hidden_size],
                'obs_dim': state_size,
                'obs_len': 1,  # Sequence length for single-step state
                'activation': nn.ReLU,
                'output_activation': nn.Tanh,
                'lambda_h': 0.05,
                'kernel_scale': 0.02,
                'norm_layer_type': "none"
            }

        self.feature_extractor = self._create_ptff(ptff_params)

        # Output layer mapping features to action probabilities
        self.output_layer = nn.Sequential(
            nn.Linear(ptff_params['sizes'][-1], action_size),
            nn.Softmax(dim=-1)
        )

        # Initialize regularization loss
        self.regularization_loss = 0

    def _create_ptff(self, params):
        """
        Create PTFF extractor

        Args:
            params (dict): Configuration parameters for the extractor

        Returns:
            SimpleNet: Fourier feature extraction network
        """

        class SimpleNet(nn.Module):
            def __init__(self, sizes, obs_dim, obs_len, activation, output_activation,
                         lambda_h, kernel_scale, norm_layer_type):
                super().__init__()
                self.lambda_h = lambda_h
                self.obs_len = obs_len
                self.obs_dim = obs_dim

                # 1D FFT filter kernel (real and imaginary parts)
                # Kernel shape: [num_freq_bins, obs_dim, 2] (2 for real/imaginary)
                num_freq_bins = self.obs_len // 2 + 1
                self.filter_kernel = nn.Parameter(
                    torch.cat([
                        torch.ones(num_freq_bins, obs_dim, 1, dtype=torch.float32),
                        torch.randn(num_freq_bins, obs_dim, 1, dtype=torch.float32) * kernel_scale
                    ], dim=2)
                )

                # MLP for feature transformation
                self.mlp = mlp(sizes, activation, output_activation)

                # Register backward hook for regularization
                self.register_full_backward_pre_hook(backward_hook)

                # Regularization loss accumulator
                self.regularization_loss = 0

            def fft_1d(self, x):
                """
                Apply 1D FFT filtering to input

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, obs_dim)

                Returns:
                    torch.Tensor: Filtered output of shape (batch_size, obs_dim)
                """
                batch_size = x.shape[0]

                # Critical fix: Reshape to (batch_size, obs_len, obs_dim) correctly
                # For obs_len=1, this becomes (batch_size, 1, obs_dim) (valid sequence length)
                x_reshaped = x.unsqueeze(1)  # Add sequence dimension at dim=1

                # Apply 1D real FFT (preserves sequence dimension structure)
                x_f = torch.fft.rfft(x_reshaped, dim=1, norm='ortho')  # Shape: (batch_size, num_freq_bins, obs_dim)

                # Convert kernel to complex: shape [num_freq_bins, obs_dim]
                kernel = torch.view_as_complex(self.filter_kernel)  # (num_freq_bins, obs_dim)

                # Critical fix: Add batch dimension to kernel for broadcasting
                kernel = kernel.unsqueeze(0)  # (1, num_freq_bins, obs_dim)

                # Element-wise multiplication (broadcasts over batch dimension)
                x_f = x_f * kernel  # Shape: (batch_size, num_freq_bins, obs_dim)

                # Inverse FFT to get back to time domain
                x_filtered = torch.fft.irfft(x_f, n=self.obs_len, dim=1, norm='ortho')  # (batch_size, obs_len, obs_dim)

                # Reshape back to original format (remove sequence dimension)
                return x_filtered.squeeze(1)  # (batch_size, obs_dim)

            def forward(self, x):
                """
                Forward pass through the feature extractor

                Args:
                    x (torch.Tensor): Input state tensor

                Returns:
                    torch.Tensor: Extracted features
                """
                # Apply FFT filtering
                x_filtered = self.fft_1d(x)

                # Pass through MLP
                features = self.mlp(x_filtered)

                # Accumulate regularization loss during training
                if self.training and features.requires_grad:
                    self.regularization_loss += self.lambda_h * (self.filter_kernel ** 2).sum()

                return features

        return SimpleNet(**params)

    def forward(self, state):
        """
        Forward pass through the actor network

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_size)

        Returns:
            torch.Tensor: Action probabilities of shape (batch_size, action_size)
        """
        features = self.feature_extractor(state)
        action_probs = self.output_layer(features)
        return action_probs

    def evaluate(self, state, epsilon=1e-6):
        """
        Evaluate state and return action, probabilities, and log probabilities

        Args:
            state (torch.Tensor): Input state tensor
            epsilon (float): Small value to avoid log(0) (default: 1e-6)

        Returns:
            tuple: (action, action_probs, log_action_probabilities)
        """
        action_probs = self.forward(state)

        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)

        # Avoid log(0) by adding epsilon to action probabilities
        log_action_probabilities = torch.log(action_probs.clamp(min=epsilon))

        # Pass regularization loss from feature extractor
        self.regularization_loss = self.feature_extractor.regularization_loss

        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_det_action(self, state):
        """
        Get deterministic action (argmax of action probabilities)

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            torch.Tensor: Deterministic action
        """
        action_probs = self.forward(state)
        # Get action with maximum probability (deterministic)
        action = torch.argmax(action_probs, dim=-1).to(state.device)

        # Pass regularization loss from feature extractor
        self.regularization_loss = self.feature_extractor.regularization_loss

        return action.detach().cpu()


# Test example
if __name__ == "__main__":
    # Set device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Hyperparameters for test
    STATE_SIZE = 19
    ACTION_SIZE = 10
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64

    # Create actor network instance
    actor = Actor(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    # Create random test states
    test_states = torch.randn(BATCH_SIZE, STATE_SIZE).to(device)

    # Test 1: Forward pass
    print("\n=== Test 1: Forward Pass ===")
    action_probs = actor(test_states)
    print(f"Input shape: {test_states.shape}")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(
        f"Probabilities sum to 1 (per sample): {torch.allclose(action_probs.sum(dim=1), torch.ones(BATCH_SIZE).to(device), atol=1e-3)}")

    # Test 2: Evaluate method
    print("\n=== Test 2: Evaluate Method ===")
    action, probs, log_probs = actor.evaluate(test_states)
    print(f"Sampled action shape: {action.shape}")
    print(f"Action probabilities shape: {probs.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Action values range: [{action.min().item()}, {action.max().item()}]")

    # Test 3: Deterministic action
    print("\n=== Test 3: Deterministic Action ===")
    det_action = actor.get_det_action(test_states)
    print(f"Deterministic action shape: {det_action.shape}")
    print(f"Deterministic action values: {det_action.unique().numpy()}")

    # Test 4: Training mode and regularization loss
    print("\n=== Test 4: Regularization Loss ===")
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # Perform a training step
    optimizer.zero_grad()
    action_probs = actor(test_states)
    dummy_loss = -torch.log(action_probs[:, 0]).mean()  # Dummy loss for training
    dummy_loss.backward()
    optimizer.step()

    print(f"Training step completed. Regularization loss after step: {actor.regularization_loss.item():.6f}")

    # Test 5: GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n=== Test 5: GPU Compatibility ===")
        actor_gpu = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).cuda()
        states_gpu = torch.randn(BATCH_SIZE, STATE_SIZE).cuda()
        action_probs_gpu = actor_gpu(states_gpu)
        print(f"GPU forward pass successful. Output shape: {action_probs_gpu.shape}")

    # Test 6: Single sample input (edge case)
    print("\n=== Test 6: Single Sample Input ===")
    single_state = torch.randn(1, STATE_SIZE).to(device)
    single_action_prob = actor(single_state)
    print(f"Single sample input shape: {single_state.shape}")
    print(f"Single sample output shape: {single_action_prob.shape}")
    print(f"Single sample probability sum: {single_action_prob.sum().item():.4f}")

    print("\nAll tests completed successfully!")
