# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    """
    A reusable linear block consisting of a linear layer, optional batch normalization, and activation.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        use_batchnorm (bool): Whether to include BatchNorm1d.
        activation (nn.Module): Activation function (e.g., LeakyReLU).
    """
    def __init__(self, in_features, out_features, use_batchnorm=False, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(LinearBlock, self).__init__()
        
        # Create a list of layers
        layers = [nn.Linear(in_features, out_features)]  # Linear transformation
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))  # Optional BatchNorm
        if activation:
            layers.append(activation)  # Optional activation function

        # Combine all layers into a sequential block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)  # Forward pass through the block

# Simple GAN Generator
class SimpleGANGenerator(nn.Module):
    """
    Generator for a simple GAN to produce embeddings from latent vectors.

    Args:
        latent_dim (int): Dimension of the latent vector (input noise).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, embedding_dim):
        super(SimpleGANGenerator, self).__init__()

        # The generator network consists of sequential LinearBlock layers
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 1024),  # First fully connected layer
            LinearBlock(1024, 512),         # Second fully connected layer
            LinearBlock(512, 256),          # Third fully connected layer
            nn.Linear(256, embedding_dim),  # Output layer
            nn.Tanh()                       # Activation: Tanh to restrict output to [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embedding.
        """
        return self.model(z)  # Pass through the model layers

# Simple GAN Discriminator
class SimpleGANDiscriminator(nn.Module):
    """
    Discriminator for a simple GAN to evaluate embeddings as real or fake.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(SimpleGANDiscriminator, self).__init__()

        # The discriminator network consists of sequential LinearBlock layers
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 256),  # First fully connected layer
            LinearBlock(256, 128),            # Second fully connected layer
            nn.Linear(128, 1),                # Output layer (real or fake)
            nn.Sigmoid()                      # Activation: Sigmoid to output [0, 1] (real/fake probability)
        )

    def forward(self, embedding):
        """
        Forward pass of the discriminator.

        Args:
            embedding (torch.Tensor): Input embedding (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(embedding)  # Pass through the model layers

# Contrastive GAN Generator
class ContrastiveGANGenerator(nn.Module):
    """
    Generator for Contrastive GAN, including batch normalization for embedding generation.

    Args:
        latent_dim (int): Dimension of the latent vector (input noise).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, embedding_dim):
        super(ContrastiveGANGenerator, self).__init__()

        # The generator network consists of sequential LinearBlock layers
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 512, use_batchnorm=True),  # First fully connected layer with batch normalization
            LinearBlock(512, 256, use_batchnorm=True),         # Second fully connected layer with batch normalization
            nn.Linear(256, embedding_dim)                      # Output layer (without activation)
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embedding.
        """
        return self.model(z)  # Pass through the model layers

# Contrastive GAN Discriminator
class ContrastiveGANDiscriminator(nn.Module):
    """
    Discriminator for Contrastive GAN to evaluate embeddings as real or fake.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(ContrastiveGANDiscriminator, self).__init__()

        # The discriminator network consists of sequential LinearBlock layers
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer with batch normalization
            LinearBlock(512, 256),            # Second fully connected layer with batch normalization
            LinearBlock(256, 128),            # Third fully connected layer with batch normalization
            nn.Linear(128, 1),                # Output layer (real or fake probability)
            nn.Sigmoid()                      # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)  # Pass through the model layers

# VAEGAN Encoder
class VAEGANEncoder(nn.Module):
    """
    Encoder for VAEGAN to map embeddings into a latent space.

    Args:
        embedding_dim (int): Dimension of the input embedding.
        latent_dim (int): Dimension of the latent vector.
    """
    def __init__(self, embedding_dim, latent_dim):
        super(VAEGANEncoder, self).__init__()

        # Encoder network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 256),
        )
        self.mu = nn.Linear(256, latent_dim)  # Mean of the latent distribution
        self.logvar = nn.Linear(256, latent_dim)  # Log variance for the latent distribution

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.
        """
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

# VAEGAN Generator
class VAEGANGenerator(nn.Module):
    """
    Generator for VAEGAN to generate embeddings from latent vectors.

    Args:
        latent_dim (int): Dimension of the latent vector.
        embedding_dim (int): Dimension of the generated embedding.
    """
    def __init__(self, latent_dim, embedding_dim):
        super(VAEGANGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 256, use_batchnorm=True),
            nn.Linear(256, embedding_dim)  # Output layer
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (from encoder).

        Returns:
            torch.Tensor: Generated embeddings.
        """
        return self.model(z)

# VAEGAN Discriminator
class VAEGANDiscriminator(nn.Module):
    """
    Discriminator for VAEGAN to evaluate embeddings as real or fake.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(VAEGANDiscriminator, self).__init__()

        # Discriminator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),
            LinearBlock(512, 256),
            LinearBlock(256, 128),
            nn.Linear(128, 1),  # Output layer (real or fake)
            nn.Sigmoid()  # Sigmoid to output a probability
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)

# WGAN Generator
class WGANGenerator(nn.Module):
    """
    Generator for WGAN to produce embeddings from latent vectors.

    Args:
        latent_dim (int): Dimension of the latent vector.
        embedding_dim (int): Dimension of the generated embedding.
    """
    def __init__(self, latent_dim, embedding_dim):
        super(WGANGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 256),
            nn.Linear(256, embedding_dim)  # Output layer (no activation for WGAN)
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embeddings.
        """
        return self.model(z)

# WGAN Critic
class WGANCritic(nn.Module):
    """
    Critic for WGAN to evaluate embeddings without sigmoid activation.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(WGANCritic, self).__init__()

        # Critic network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),
            LinearBlock(512, 256),
            nn.Linear(256, 1)  # Output scalar without activation (no sigmoid)
        )

    def forward(self, x):
        """
        Forward pass through the critic.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Scalar output for the input (real/fake score).
        """
        return self.model(x)

# Cross-Domain Embedding Learning Generators and Discriminators
class CrossDomainGenerator(nn.Module):
    """
    Generator for cross-domain embedding learning tasks.

    Args:
        latent_dim (int): Dimension of the latent vector (input noise).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, embedding_dim):
        super(CrossDomainGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),         # Second fully connected layer
            nn.Linear(256, embedding_dim)  # Output layer
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embedding.
        """
        return self.model(z)

class CrossDomainDiscriminator(nn.Module):
    """
    Discriminator for cross-domain embedding learning tasks.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(CrossDomainDiscriminator, self).__init__()

        # Discriminator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),            # Second fully connected layer
            nn.Linear(256, 1),                # Output layer (real or fake)
            nn.Sigmoid()                      # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)

# CycleGAN Generators
class CycleGenerator(nn.Module):
    """
    Generator for CycleGAN to map embeddings between domains.

    Args:
        embedding_dim (int): Dimension of the input and output embeddings.
    """
    def __init__(self, embedding_dim):
        super(CycleGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),            # Second fully connected layer
            nn.Linear(256, embedding_dim)     # Output layer (same dimensionality for A → B and B → A)
        )

    def forward(self, x):
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input embeddings (from domain A or domain B).

        Returns:
            torch.Tensor: Transformed embeddings (from domain B or domain A).
        """
        return self.model(x)

class CycleDiscriminator(nn.Module):
    """
    Discriminator for CycleGAN to distinguish between real and fake embeddings.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(CycleDiscriminator, self).__init__()

        # Discriminator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),            # Second fully connected layer
            nn.Linear(256, 1),                # Output layer (real or fake)
            nn.Sigmoid()                      # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)

# Dual GAN Framework Generators and Discriminators
class DualGANGenerator(nn.Module):
    """
    Generator for Dual GAN to map embeddings between two domains.

    Args:
        latent_dim (int): Dimension of the latent vector (input noise).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, embedding_dim):
        super(DualGANGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),         # Second fully connected layer
            nn.Linear(256, embedding_dim)  # Output layer
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embedding.
        """
        return self.model(z)

class DualGANDiscriminator(nn.Module):
    """
    Discriminator for Dual GAN to evaluate embeddings for two domains.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(DualGANDiscriminator, self).__init__()

        # Discriminator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer
            LinearBlock(512, 256),            # Second fully connected layer
            nn.Linear(256, 1),                # Output layer (real or fake)
            nn.Sigmoid()                      # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)

# Contrastive-Guided Dual-GAN Framework Generators and Discriminators
class ContrastiveDualGANGenerator(nn.Module):
    """
    Generator for Contrastive-Guided Dual GAN with batch normalization.

    Args:
        latent_dim (int): Dimension of the latent vector (input noise).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, embedding_dim):
        super(ContrastiveDualGANGenerator, self).__init__()

        # Generator network
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 512, use_batchnorm=True),  # First fully connected layer with batch normalization
            LinearBlock(512, 256, use_batchnorm=True),         # Second fully connected layer with batch normalization
            nn.Linear(256, embedding_dim)                      # Output layer
        )

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Latent vector (random noise).

        Returns:
            torch.Tensor: Generated embedding.
        """
        return self.model(z)

class ContrastiveDualGANDiscriminator(nn.Module):
    """
    Discriminator for Contrastive-Guided Dual GAN with contrastive learning.

    Args:
        embedding_dim (int): Dimension of the input embedding.
    """
    def __init__(self, embedding_dim):
        super(ContrastiveDualGANDiscriminator, self).__init__()

        # Discriminator network
        self.model = nn.Sequential(
            LinearBlock(embedding_dim, 512),  # First fully connected layer with batch normalization
            LinearBlock(512, 256),            # Second fully connected layer with batch normalization
            nn.Linear(256, 1),                # Output layer (real or fake)
            nn.Sigmoid()                      # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        return self.model(x)

# NT-Xent Loss
def nt_xent_loss(embeddings, temperature=0.5):
    """
    Computes the NT-Xent Loss for contrastive learning.

    Args:
        embeddings (torch.Tensor): Input embeddings.
        temperature (float): Temperature parameter for scaling similarities.

    Returns:
        torch.Tensor: Computed NT-Xent loss.
    """
    batch_size = embeddings.shape[0]

    # Calculate pairwise cosine similarity
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Numerical stability: subtract the max similarity from each element
    sim = sim - torch.max(sim, 1)[0].view(batch_size, 1)

    # Mask the diagonal to avoid comparing embeddings with themselves
    mask = torch.eye(batch_size, device=embeddings.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))

    # Create labels for contrastive loss (diagonal ones)
    labels = torch.arange(batch_size, device=embeddings.device)

    # Cross-entropy loss between the similarity matrix and the labels
    return F.cross_entropy(sim, labels)

# Gradient Penalty for WGAN-GP
def compute_gradient_penalty(model, real_samples, fake_samples, device):
    """
    Computes the gradient penalty for WGAN-GP.

    Args:
        model (nn.Module): The critic or discriminator.
        real_samples (torch.Tensor): Real data samples.
        fake_samples (torch.Tensor): Fake data samples.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: Computed gradient penalty.
    """
    batch_size = real_samples.size(0)

    # Randomly interpolate between real and fake samples
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Forward pass through the critic (model)
    interpolated_pred = model(interpolated)

    # Compute gradients w.r.t. interpolated samples
    grad_outputs = torch.ones(interpolated_pred.size(), device=device)
    grad_interpolated = torch.autograd.grad(
        outputs=interpolated_pred,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the L2 norm of the gradients
    grad_norm = grad_interpolated.view(batch_size, -1).norm(2, dim=1)

    # Compute the gradient penalty (difference between norm and 1)
    return torch.mean((grad_norm - 1) ** 2)

# New Proposed Model: Semi-Supervised GAN
class SemiSupervisedGANDiscriminator(nn.Module):
    """
    Discriminator for Semi-Supervised GAN with classification and adversarial outputs.

    Args:
        embedding_dim (int): Dimension of the input embedding.
        num_classes (int): Number of classes for classification.
    """
    def __init__(self, embedding_dim, num_classes):
        super(SemiSupervisedGANDiscriminator, self).__init__()

        # Shared feature extraction network
        self.shared_model = nn.Sequential(
            LinearBlock(embedding_dim, 512),
            LinearBlock(512, 256),
        )

        # Adversarial head (real/fake)
        self.adv_head = nn.Sequential(
            LinearBlock(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Classification head (real classes)
        self.class_head = nn.Sequential(
            LinearBlock(256, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the Semi-Supervised Discriminator.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            tuple: Adversarial output (real/fake probability) and class prediction logits.
        """
        shared = self.shared_model(x)
        adv_output = self.adv_head(shared)
        class_output = self.class_head(shared)
        return adv_output, class_output

import torch.nn.utils as nn_utils

class ConditionalGANGenerator(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_classes):
        super(ConditionalGANGenerator, self).__init__()
        self.num_classes = num_classes  # Store num_classes for one-hot encoding
        self.model = nn.Sequential(
            nn_utils.spectral_norm(nn.Linear(latent_dim + num_classes, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn_utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, z, labels):
        if labels.ndimension() == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        elif labels.ndimension() == 2 and labels.size(1) != self.num_classes:
            raise ValueError(f"Expected labels with size (batch_size, {self.num_classes}), but got {labels.size()}.")
        z = torch.cat([z, labels], dim=1)
        return self.model(z)

# Conditional GAN Discriminator
class ConditionalGANDiscriminator(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ConditionalGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn_utils.spectral_norm(nn.Linear(embedding_dim + num_classes, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn_utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        if labels.ndimension() == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        elif labels.ndimension() == 2 and labels.size(1) != self.num_classes:
            raise ValueError(f"Expected labels with size (batch_size, {self.num_classes}), but got {labels.size()}.")
        x = torch.cat([x, labels], dim=1)
        return self.model(x)


class ConditionalGANDiscriminator(nn.Module):
    """
    Discriminator for Conditional GAN (cGAN) to evaluate embeddings conditioned on labels.

    Args:
        embedding_dim (int): Dimension of the input embedding.
        num_classes (int): Number of classes for conditional input.
    """
    def __init__(self, embedding_dim, num_classes):
        super(ConditionalGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn_utils.spectral_norm(nn.Linear(embedding_dim + num_classes, 512)),  # Concatenate embedding and label
            nn.LeakyReLU(0.2, inplace=True),
            nn_utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).
            labels (torch.Tensor): Conditional labels (e.g., one-hot encoded labels).

        Returns:
            torch.Tensor: Probability of the input being real (between 0 and 1).
        """
        x = torch.cat([x, labels], dim=1)  # Concatenate embedding and labels
        return self.model(x)

# InfoGAN Generator
# class InfoGANGenerator(nn.Module):
#     """
#     Generator for InfoGAN to produce embeddings from latent vectors with categorical and continuous latent variables.

#     Args:
#         latent_dim (int): Dimension of the continuous latent vector (input noise).
#         categorical_dim (int): Dimension of the categorical latent vector (e.g., one-hot encoded class labels).
#         embedding_dim (int): Dimension of the generated embedding (output).
#     """
#     def __init__(self, latent_dim, categorical_dim, embedding_dim):
#         super(InfoGANGenerator, self).__init__()

#         self.model = nn.Sequential(
#             nn_utils.spectral_norm(nn.Linear(latent_dim + categorical_dim, 512)),  # Latent + categorical variables as input
#             nn.LeakyReLU(0.2, inplace=True),
#             nn_utils.spectral_norm(nn.Linear(512, 256)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, embedding_dim)
#         )

#     def forward(self, z, c):
#         """
#         Forward pass through the generator.

#         Args:
#             z (torch.Tensor): Continuous latent vector (random noise).
#             c (torch.Tensor): Categorical latent vector (one-hot encoded labels).

#         Returns:
#             torch.Tensor: Generated embedding conditioned on the latent variables.
#         """
#         # Ensure categorical vector `c` has the correct shape (batch_size, categorical_dim)
#         if c.ndimension() == 1:  # If `c` is a 1D vector (e.g., class indices), make it one-hot
#             c = F.one_hot(c, num_classes=self.categorical_dim).float()  # Convert to one-hot

#         z = torch.cat([z, c], dim=1)  # Concatenate continuous and categorical latent vectors
#         return self.model(z)

class InfoGANGenerator(nn.Module):
    """
    Generator for InfoGAN to produce embeddings from latent vectors with categorical and continuous latent variables.

    Args:
        latent_dim (int): Dimension of the continuous latent vector (input noise).
        categorical_dim (int): Dimension of the categorical latent vector (e.g., one-hot encoded class labels).
        embedding_dim (int): Dimension of the generated embedding (output).
    """
    def __init__(self, latent_dim, categorical_dim, embedding_dim):
        super(InfoGANGenerator, self).__init__()

        self.categorical_dim = categorical_dim  # Store categorical_dim as an attribute

        self.model = nn.Sequential(
            nn_utils.spectral_norm(nn.Linear(latent_dim + categorical_dim, 512)),  # Latent + categorical variables as input
            nn.LeakyReLU(0.2, inplace=True),
            nn_utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, z, c):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Continuous latent vector (random noise).
            c (torch.Tensor): Categorical latent vector (one-hot encoded labels).

        Returns:
            torch.Tensor: Generated embedding conditioned on the latent variables.
        """
        # Ensure categorical vector `c` has the correct shape (batch_size, categorical_dim)
        if c.ndimension() == 1:  # If `c` is a 1D vector (e.g., class indices), make it one-hot
            c = F.one_hot(c, num_classes=self.categorical_dim).float()  # Convert to one-hot

        z = torch.cat([z, c], dim=1)  # Concatenate continuous and categorical latent vectors
        return self.model(z)

# InfoGAN Discriminator
class InfoGANDiscriminator(nn.Module):
    """
    Discriminator for InfoGAN to evaluate embeddings and predict both real/fake and categorical information.

    Args:
        embedding_dim (int): Dimension of the input embedding.
        categorical_dim (int): Number of categories for the categorical variable.
    """
    def __init__(self, embedding_dim, categorical_dim):
        super(InfoGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn_utils.spectral_norm(nn.Linear(embedding_dim, 512)),  # First fully connected layer
            nn.LeakyReLU(0.2, inplace=True),
            nn_utils.spectral_norm(nn.Linear(512, 256)),            # Second fully connected layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),                                        # Output layer (real or fake)
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(256, categorical_dim)  # Categorical classifier head (uses categorical_dim)

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input embeddings (real or fake).

        Returns:
            tuple: Adversarial output (real/fake probability) and class prediction logits.
        """
        features = self.model[0:3](x)
        real_or_fake = self.model[3](features)
        class_preds = self.classifier(features)
        return real_or_fake, class_preds

