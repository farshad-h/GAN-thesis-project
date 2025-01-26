# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA, TruncatedSVD, NMF, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def init_weights(m):
    """
    Initializes the weights of the given layer.

    Method:
    - Uses Xavier/Glorot initialization for Conv2d, ConvTranspose2d, and Linear layers.
    - Sets bias to zero if it exists.

    Args:
        m (nn.Module): Layer to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # Xavier/Glorot weight initialization for better convergence
        nn.init.xavier_uniform_(m.weight)
        # Initialize bias to zero if it exists
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class BasicAutoencoder(nn.Module):
    """
    A simple autoencoder for MNIST.

    Architecture:
    - Encoder: Two convolutional layers followed by max-pooling.
    - Decoder: Two transposed convolutional layers to reconstruct the input.

    Suitable for learning compact embeddings of MNIST digits.
    """
    def __init__(self, code_dim):
        super(BasicAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input: (1, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: (16, 14, 14)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # Output: (8, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: (8, 7, 7)
        )
        self.fc_encoder = nn.Linear(8 * 7 * 7, code_dim)  # Flatten to embedding dimension

        # Decoder
        self.fc_decoder = nn.Linear(code_dim, 8 * 7 * 7)  # Unflatten to feature map
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (8, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 28, 28)
            nn.Tanh()  # Outputs pixel values in range [-1, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, -1)  # Flatten to (batch_size, 8 * 7 * 7)
        encoded = self.fc_encoder(encoded)  # Output: (batch_size, code_dim)

        # Decoder
        decoded = self.fc_decoder(encoded)
        decoded = self.decoder(decoded)  # Output: (batch_size, 1, 28, 28)

        return encoded, decoded

class IntermediateAutoencoder(nn.Module):
    """
    An autoencoder with intermediate complexity for MNIST.

    Features:
    - Uses Batch Normalization to improve stability during training.
    - Deeper architecture compared to BasicAutoencoder, with more feature maps.

    Designed for more expressive embeddings while retaining simplicity.
    """
    def __init__(self, code_dim):
        super(IntermediateAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: (1, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # Output: (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 14, 14)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Output: (64, 7, 7)
        )
        self.fc_encoder = nn.Linear(64 * 7 * 7, code_dim)  # Flatten to embedding dimension

        # Decoder
        self.fc_decoder = nn.Linear(code_dim, 64 * 7 * 7)  # Unflatten to feature map
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, -1)  # Flatten to (batch_size, 64 * 7 * 7)
        encoded = self.fc_encoder(encoded)  # Output: (batch_size, code_dim)

        # Decoder
        decoded = self.fc_decoder(encoded)
        decoded = self.decoder(decoded)  # Output: (batch_size, 1, 28, 28)

        return encoded, decoded

class EnhancedAutoencoder(nn.Module):
    """
    A deep autoencoder with advanced reconstruction capabilities for MNIST.

    Features:
    - Deeper architecture with additional convolutional and transposed convolutional layers.
    - Utilizes Batch Normalization and LeakyReLU activations.
    - Capable of learning highly expressive embeddings.

    Designed for datasets requiring intricate reconstructions.
    """
    def __init__(self, code_dim):
        super(EnhancedAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: (1, 28, 28)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # Output: (32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 14, 14)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Output: (64, 7, 7)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: (128, 7, 7)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # Output: (128, 3, 3)
        )
        self.fc_encoder = nn.Linear(128 * 3 * 3, code_dim)  # Flatten to embedding dimension

        # Decoder
        self.fc_decoder = nn.Linear(code_dim, 128 * 3 * 3)  # Unflatten to feature map
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (128, 3, 3)),  # Reshape to (batch_size, 128, 3, 3)

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 7, 7)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=1),  # Output: (32, 14, 14)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 28, 28)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, -1)  # Flatten to (batch_size, 128 * 3 * 3)
        encoded = self.fc_encoder(encoded)  # Output: (batch_size, code_dim)

        # Decoder
        decoded = self.fc_decoder(encoded)
        decoded = self.decoder(decoded)  # Output: (batch_size, 1, 28, 28)

        return encoded, decoded

class AdvancedAutoencoder(nn.Module):
    """
    A more advanced autoencoder with skip connections.

    Features:
    - Skip connections between encoder and decoder layers for better gradient flow.
    - LeakyReLU activations and Batch Normalization for improved performance.

    Suitable for complex embedding tasks requiring detailed reconstruction.
    """
    def __init__(self, code_dim):
        super(AdvancedAutoencoder, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: (1, 28, 28)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)  # Output: (32, 14, 14)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 14, 14)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # Output: (64, 7, 7)
        )
        self.fc_encoder = nn.Linear(64 * 7 * 7, code_dim)  # Flatten to embedding dimension

        # Decoder with Skip Connections
        self.fc_decoder = nn.Linear(code_dim, 64 * 7 * 7)  # Unflatten to feature map
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Learned upsampling
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        batch_size = x.size(0)

        # First encoder layer
        encoded1 = self.encoder_conv1(x)  # Output: (32, 14, 14)

        # Second encoder layer
        encoded2 = self.encoder_conv2(encoded1)  # Output: (64, 7, 7)

        # Flatten and encode
        encoded2_flat = encoded2.view(batch_size, -1)  # Flatten to (batch_size, 64 * 7 * 7)
        encoded = self.fc_encoder(encoded2_flat)  # Output: (batch_size, code_dim)

        # Decoder
        decoded_fc = self.fc_decoder(encoded)  # Unflatten to (batch_size, 64 * 7 * 7)
        decoded_fc = decoded_fc.view(batch_size, 64, 7, 7)  # Reshape to (batch_size, 64, 7, 7)

        # First decoder layer with skip connection
        decoded1 = self.decoder_conv1(decoded_fc)  # Output: (32, 14, 14)
        decoded1 = decoded1 + encoded1  # Skip connection from encoder_conv1

        # Second decoder layer
        decoded = self.decoder_conv2(decoded1)  # Output: (1, 28, 28)

        return encoded, decoded

class BasicVAE(nn.Module):
    """
    A simple Variational Autoencoder (VAE) for MNIST.
    Assumes input images are normalized to [-1, 1].

    Args:
        input_dim (int): Number of input channels (e.g., 1 for grayscale images).
        code_dim (int): Dimensionality of the latent space.

    Forward Returns:
        mu (torch.Tensor): Mean of the latent space distribution.
        logvar (torch.Tensor): Log variance of the latent space distribution.
        decoded (torch.Tensor): Reconstructed output image.
    """
    def __init__(self, input_dim=1, code_dim=32):
        super(BasicVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # 16x14x14
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),  # 8x7x7
            nn.Flatten()  # 8*7*7 = 392
        )

        self.fc_mu = nn.Linear(392, code_dim)  # Latent space mean
        self.fc_logvar = nn.Linear(392, code_dim)  # Latent space log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 392),
            nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),  # 8x7x7
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_dim, kernel_size=3, stride=1, padding=1),  # 1x28x28
            nn.Tanh()  # Output in [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.

        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 28, 28).

        Returns:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.
            decoded (torch.Tensor): Reconstructed output image.
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return mu, logvar, decoded

class ImprovedVAE(nn.Module):
    """
    An improved VAE with a bottleneck layer for MNIST.
    Assumes input images are normalized to [-1, 1].

    Args:
        input_dim (int): Number of input channels (e.g., 1 for grayscale images).
        code_dim (int): Dimensionality of the latent space.

    Forward Returns:
        mu (torch.Tensor): Mean of the latent space distribution.
        logvar (torch.Tensor): Log variance of the latent space distribution.
        decoded (torch.Tensor): Reconstructed output image.
    """
    def __init__(self, input_dim=1, code_dim=64):
        super(ImprovedVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 64x7x7
            nn.Flatten(),  # 64*7*7 = 3136
            nn.Linear(3136, 256),  # Bottleneck layer
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, code_dim)  # Latent space mean
        self.fc_logvar = nn.Linear(256, code_dim)  # Latent space log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3136),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # 64x7x7
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_dim, kernel_size=4, stride=2, padding=1),  # 1x28x28
            nn.Tanh()  # Output in [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.

        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 28, 28).

        Returns:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.
            decoded (torch.Tensor): Reconstructed output image.
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return mu, logvar, decoded

class FlexibleVAE(nn.Module):
    """
    A flexible VAE with dynamic input shapes and optional projection head.
    Assumes input images are normalized to [-1, 1].

    Args:
        input_shape (tuple): Shape of the input data (e.g., (1, 28, 28) for MNIST).
        code_dim (int): Dimensionality of the latent space.
        projection_dim (int, optional): Dimensionality of the projection head.

    Forward Returns:
        mu (torch.Tensor): Mean of the latent space distribution.
        logvar (torch.Tensor): Log variance of the latent space distribution.
        decoded (torch.Tensor): Reconstructed output image.
    """
    def __init__(self, input_shape=(1, 28, 28), code_dim=32, projection_dim=None):
        super(FlexibleVAE, self).__init__()
        self.input_shape = input_shape
        self.flat_size = np.prod(input_shape)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, code_dim)  # Latent space mean
        self.fc_logvar = nn.Linear(256, code_dim)  # Latent space log variance

        # Optional projection head
        if projection_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(code_dim, projection_dim),
                nn.ReLU()
            )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_size),
            nn.Tanh()  # Output in [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.

        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, *input_shape).

        Returns:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.
            decoded (torch.Tensor): Reconstructed output image.
        """
        x = x.view(-1, self.flat_size)  # Flatten input
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        if hasattr(self, 'projection_head'):
            z = self.projection_head(z)
        decoded = self.decoder(z)
        decoded = decoded.view(-1, *self.input_shape)  # Reshape output
        return mu, logvar, decoded

class ImprovedFlexibleVAE(nn.Module):
    """
    A flexible and improved VAE with convolutional encoder layers.
    Assumes input images are normalized to [-1, 1].

    Args:
        input_shape (tuple): Shape of the input data (e.g., (1, 28, 28) for MNIST).
        code_dim (int): Dimensionality of the latent space.
        projection_dim (int, optional): Dimensionality of the projection head.

    Forward Returns:
        mu (torch.Tensor): Mean of the latent space distribution.
        logvar (torch.Tensor): Log variance of the latent space distribution.
        decoded (torch.Tensor): Reconstructed output image.
    """
    def __init__(self, input_shape=(1, 28, 28), code_dim=64, projection_dim=None):
        super(ImprovedFlexibleVAE, self).__init__()
        self.input_shape = input_shape
        self.flat_size = np.prod(input_shape)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 64x7x7
            nn.Flatten(),  # 64*7*7 = 3136
            nn.Linear(3136, 256),  # Bottleneck layer
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, code_dim)  # Latent space mean
        self.fc_logvar = nn.Linear(256, code_dim)  # Latent space log variance

        # Optional projection head
        if projection_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(code_dim, projection_dim),
                nn.ReLU()
            )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3136),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # 64x7x7
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x28x28
            nn.Tanh()  # Output in [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.

        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, *input_shape).

        Returns:
            mu (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.
            decoded (torch.Tensor): Reconstructed output image.
        """
        x = x.view(-1, *self.input_shape)  # Ensure input matches expected shape
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        if hasattr(self, 'projection_head'):
            z = self.projection_head(z)
        decoded = self.decoder(z)
        return mu, logvar, decoded

class DenoisingAutoencoder(nn.Module):
    """
    A denoising autoencoder specifically designed for the MNIST dataset.

    Features:
    - Encoder: Convolutional layers with max-pooling for feature extraction.
    - Decoder: Transposed convolutional layers for reconstructing clean inputs.
    - Optional Projection Head: Supports embedding projection for contrastive learning.

    Args:
        code_dim (int): Dimensionality of the latent space.
        projection_dim (int, optional): Dimensionality of the projection head.
        strong_architecture (bool): Whether to use a deeper encoder/decoder structure.

    Methods:
        forward(x): Encodes the noisy input and reconstructs the clean output.
    """

    def __init__(self, code_dim=32, projection_dim=None, strong_architecture=False):
        super(DenoisingAutoencoder, self).__init__()
        self.code_dim = code_dim
        self.projection_dim = projection_dim
        self.strong_architecture = strong_architecture

        # Encoder
        self.encoder = self._build_encoder()

        # Latent space mapping
        self.fc_encoder = nn.Linear(self._get_encoder_output_size(), code_dim)

        # Optional Projection Head
        if projection_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(code_dim, projection_dim),
                nn.ReLU()
            )
        else:
            self.projection_head = None

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """
        Build the encoder based on the architecture choice (strong or basic).

        Returns:
            nn.Sequential: Encoder network.
        """
        if self.strong_architecture:
            return nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 32x28x28
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, 2),  # 32x14x14
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32x14x14 -> 64x14x14
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),  # 64x7x7
                nn.Flatten()  # 64x7x7 -> 3136
            )
        else:
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2, 2),  # 16x14x14
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 8x14x14
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2, 2),  # 8x7x7
                nn.Flatten()  # 8x7x7 -> 392
            )

    def _get_encoder_output_size(self):
        """
        Compute the output size of the encoder.

        Returns:
            int: Number of features after the encoder.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST input shape
            dummy_output = self.encoder(dummy_input)
            return dummy_output.shape[1]

    def _build_decoder(self):
        """
        Build the decoder based on the architecture choice (strong or basic).

        Returns:
            nn.Sequential: Decoder network.
        """
        if self.strong_architecture:
            return nn.Sequential(
                nn.Linear(self.code_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64 * 7 * 7),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),  # 64x7x7
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x28x28
                nn.Tanh()  # Output in [-1, 1]
            )
        else:
            return nn.Sequential(
                nn.Linear(self.code_dim, 8 * 7 * 7),
                nn.ReLU(),
                nn.Unflatten(1, (8, 7, 7)),  # 8x7x7
                nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x14x14
                nn.ReLU(),
                nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x28x28
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 1x28x28
                nn.Tanh()  # Output in [-1, 1]
            )

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        encoded = encoded.view(x.size(0), -1)  # Flatten
        encoded = self.fc_encoder(encoded)  # Map to latent space

        # Decoding
        decoded = self.decoder(encoded)

        # Return projected_encoded only if projection_head is defined
        if self.projection_head:
            projected_encoded = self.projection_head(encoded)
            return projected_encoded, encoded, decoded
        else:
            return None, encoded, decoded

def apply_dimensionality_reduction(method, data, n_components, scaler=None, **kwargs):
    """
    Apply dimensionality reduction to input data.

    Args:
        method (class): Dimensionality reduction class (e.g., PCA, TruncatedSVD, NMF).
        data (np.ndarray): Input data to reduce dimensions.
        n_components (int): Number of components to reduce to.
        scaler (object, optional): Scaler to preprocess data (e.g., StandardScaler, MinMaxScaler).
        **kwargs: Additional arguments for the dimensionality reduction method.

    Returns:
        np.ndarray: Reduced dimensional data.
    """
    if scaler:
        data = scaler.fit_transform(data)
    model = method(n_components=n_components, **kwargs)
    return model.fit_transform(data)


def process_matrix_factorization(sampled_x, sampled_y, n_components=50):
    """
    Apply matrix factorization methods (PCA, SVD, NMF) to input data.

    Args:
        sampled_x (np.ndarray): Input data.
        sampled_y (np.ndarray): Corresponding labels.
        n_components (int): Number of components for factorization.

    Returns:
        dict: Dictionary containing embeddings for each method.
        torch.Tensor: Tensor of labels.
    """
    results = {}
    sampled_x_flat = sampled_x.reshape(sampled_x.shape[0], -1)

    # Standardize and Min-Max scale
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    # PCA
    pca_embeddings = apply_dimensionality_reduction(PCA, sampled_x_flat, n_components=n_components, scaler=scaler_standard)
    results["PCA"] = torch.tensor(pca_embeddings, dtype=torch.float32)

    # SVD
    svd_embeddings = apply_dimensionality_reduction(TruncatedSVD, sampled_x_flat, n_components=n_components, scaler=scaler_standard)
    results["SVD"] = torch.tensor(svd_embeddings, dtype=torch.float32)

    # NMF
    nmf_embeddings = apply_dimensionality_reduction(NMF, sampled_x_flat, n_components=n_components, scaler=scaler_minmax)
    results["NMF"] = torch.tensor(nmf_embeddings, dtype=torch.float32)

    return results, torch.tensor(sampled_y, dtype=torch.long)

def apply_sift(data, n_features=50):
    """
    Applies SIFT (Scale-Invariant Feature Transform) to extract features from images.

    Args:
        data (np.ndarray): Array of grayscale images with shape (N, H, W).
        n_features (int): Maximum number of features to extract per image.

    Returns:
        np.ndarray: Extracted SIFT features with shape (N, n_features * 128).
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    descriptors = []

    for image in data:
        image = (image * 255).astype(np.uint8)  # Convert image to uint8 format
        keypoints, des = sift.detectAndCompute(image, None)
        if des is not None:
            # Truncate or pad descriptors to ensure consistent shape
            if len(des) > n_features:
                des = des[:n_features]
            else:
                des = np.pad(des, ((0, n_features - len(des)), (0, 0)), mode='constant')
            descriptors.append(des.flatten())  # Flatten to (n_features * 128,)
        else:
            # If no keypoints are found, use zeros
            descriptors.append(np.zeros(n_features * 128))

    return np.array(descriptors)

def process_feature_extraction(sampled_x, sampled_y, n_features=50, kernel="rbf", n_components=50):
    """
    Extracts features using SIFT and Kernel PCA from the input data.

    Args:
        sampled_x (np.ndarray): Input data with shape (N, H, W) for images.
        sampled_y (np.ndarray): Corresponding labels for input data.
        n_features (int): Number of features to extract using SIFT.
        kernel (str): Kernel type for Kernel PCA (e.g., "rbf", "linear").
        n_components (int): Number of components for Kernel PCA.

    Returns:
        dict: Dictionary containing extracted features (e.g., "SIFT", "Kernel PCA").
        torch.Tensor: Tensor of labels corresponding to the input data.
    """
    results = {}

    # SIFT Feature Extraction
    sift_features = apply_sift(sampled_x, n_features=n_features)
    results["SIFT"] = torch.tensor(sift_features, dtype=torch.float32)

    # Kernel PCA Feature Extraction
    sampled_x_flat = sampled_x.reshape(sampled_x.shape[0], -1)
    scaler_standard = StandardScaler()
    sampled_x_scaled = scaler_standard.fit_transform(sampled_x_flat)
    kpca_embeddings = apply_dimensionality_reduction(
        KernelPCA, sampled_x_scaled, n_components=n_components, kernel=kernel
    )
    results["Kernel PCA"] = torch.tensor(kpca_embeddings, dtype=torch.float32)

    return results, torch.tensor(sampled_y, dtype=torch.long)

class FlowLayer(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes a single Normalizing Flow layer.

        Args:
            input_dim (int): Dimensionality of the input.
        """
        super(FlowLayer, self).__init__()
        self.scale = nn.Linear(input_dim, input_dim)
        self.shift = nn.Linear(input_dim, input_dim)

    def forward(self, z):
        """
        Forward pass for the flow layer.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
            torch.Tensor: Log determinant of the Jacobian.
        """
        scale = torch.tanh(self.scale(z))
        shift = self.shift(z)
        transformed_z = z * torch.exp(scale) + shift
        log_det_jacobian = torch.sum(scale, dim=-1)
        return transformed_z, log_det_jacobian


class NormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, num_flows):
        """
        Initializes a Normalizing Flow model with multiple layers.

        Args:
            input_dim (int): Dimensionality of the input.
            num_flows (int): Number of flow layers.
        """
        super(NormalizingFlowModel, self).__init__()
        self.flows = nn.ModuleList([FlowLayer(input_dim) for _ in range(num_flows)])

    def forward(self, z):
        """
        Forward pass through all flow layers.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
            torch.Tensor: Total log determinant of the Jacobian.
        """
        log_det_jacobian_total = torch.zeros(z.size(0), device=z.device)
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_total += log_det_jacobian
        return z, log_det_jacobian_total


def log_prob(z, model, base_distribution, detailed=False):
    """
    Computes the log probability of a tensor under the model and base distribution.

    Args:
        z (torch.Tensor): Input tensor.
        model (NormalizingFlowModel): Normalizing Flow model.
        base_distribution (torch.distributions.Distribution): Base distribution.
        detailed (bool, optional): If True, returns detailed outputs including transformed
                                   tensor, base log probability, and log determinant of Jacobian.

    Returns:
        torch.Tensor: Log probability.
        If detailed=True:
            torch.Tensor: Transformed tensor.
            torch.Tensor: Log probability from the base distribution.
            torch.Tensor: Log determinant of the Jacobian.
    """
    z_transformed, log_det_jacobian = model(z)
    log_base_prob = base_distribution.log_prob(z_transformed).sum(dim=-1)
    if detailed:
        return z_transformed, log_base_prob, log_det_jacobian
    return log_base_prob + log_det_jacobian


def train_nf_model(model, embeddings, num_epochs=10, lr=1e-3, batch_size=128, save_path=None, device=None):
    """
    Trains a Normalizing Flow model.

    Args:
        model (NormalizingFlowModel): Normalizing Flow model to train.
        embeddings (torch.Tensor): Input embeddings for training.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        save_path (str, optional): Path to save the model checkpoints. Defaults to None.
        device (torch.device, optional): Device to use (e.g., 'cuda' or 'cpu'). Defaults to None.

    Returns:
        NormalizingFlowModel: Trained Normalizing Flow model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    embeddings = embeddings.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    base_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(embeddings.size(1), device=device),
        torch.eye(embeddings.size(1), device=device)
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            batch_z = batch[0].to(device)
            loss = -torch.mean(log_prob(batch_z, model, base_distribution))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

        if save_path:
            torch.save(model.state_dict(), save_path)

    return model


def refine_embeddings_NF(embedding_type, embedding_method, embeddings, num_flows=4, num_epochs=20):
    """
    Refines embeddings using a Normalizing Flow model.

    Args:
        embedding_type (str): Type of embedding (e.g., 'Matrix Factorization').
        embedding_method (str): Specific embedding method (e.g., 'PCA').
        embeddings (dict): Dictionary containing all embeddings.
        num_flows (int): Number of flow layers in the model.
        num_epochs (int): Number of training epochs.

    Returns:
        dict: Updated embeddings dictionary with refined embeddings.
    """
    embeddings_tensor = torch.tensor(embeddings[embedding_type][embedding_method]).float()
    latent_dim = embeddings_tensor.shape[1]
    nf_model = NormalizingFlowModel(input_dim=latent_dim, num_flows=num_flows)
    trained_nf_model = train_nf_model(nf_model, embeddings_tensor, num_epochs=num_epochs)

    with torch.no_grad():
        refined_embeddings, _ = trained_nf_model(embeddings_tensor)
        refined_key = f"Refined {embedding_method} Embeddings"
        embeddings["Generative Models"][refined_key] = refined_embeddings.numpy()

    return embeddings

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim, use_batchnorm=True):
        """
        Initializes the projection head for SimCLR with optional batch normalization.

        Args:
            input_dim (int): Input dimensionality.
            projection_dim (int): Dimensionality of the projected space.
            use_batchnorm (bool, optional): Whether to use BatchNorm. Defaults to True.
        """
        super(ProjectionHead, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.fc1 = nn.Linear(input_dim, projection_dim)
        self.bn1 = nn.BatchNorm1d(projection_dim) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(projection_dim, projection_dim)
        self.bn2 = nn.BatchNorm1d(projection_dim) if use_batchnorm else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the projection head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected tensor.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=128, temperature=0.1):
        """
        Initializes the SimCLR model.

        Args:
            base_model (nn.Module): Base encoder model with an 'output_dim' attribute.
            projection_dim (int): Dimensionality of the projection head.
            temperature (float): Temperature scaling for contrastive loss.
        """
        super(SimCLR, self).__init__()
        assert hasattr(base_model, 'output_dim'), "Base model must have 'output_dim' attribute."
        self.encoder = base_model
        self.projection_head = ProjectionHead(input_dim=base_model.output_dim, projection_dim=projection_dim)
        self.temperature = temperature

    def forward(self, x1, x2):
        """
        Forward pass through the encoder and projection head.

        Args:
            x1 (torch.Tensor): First augmented view of the input.
            x2 (torch.Tensor): Second augmented view of the input.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Encode and project both views
        h1 = self.projection_head(self.encoder(x1))
        h2 = self.projection_head(self.encoder(x2))

        # Compute contrastive loss
        loss = self.contrastive_loss(h1, h2)
        return loss

    def contrastive_loss(self, h1, h2):
        """
        Computes contrastive loss for the SimCLR model.

        Args:
            h1 (torch.Tensor): Projections from the first augmented view.
            h2 (torch.Tensor): Projections from the second augmented view.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Concatenate projections
        projections = torch.cat([h1, h2], dim=0)
        projections = F.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(similarity_matrix.size(0), device=projections.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # Compute loss
        labels = torch.arange(projections.size(0), device=projections.device)
        labels = (labels + projections.size(0) // 2) % projections.size(0)  # Shift labels for positive pairs
        return F.cross_entropy(similarity_matrix, labels)