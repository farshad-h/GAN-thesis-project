# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize, hflip

def augment(images):
    """
    Apply tensor-based augmentations to images.

    Args:
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: Augmented images.
    """
    # Resize and crop
    images = resize(images, size=[28, 28])
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        images = hflip(images)
    return images

class VicRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (VicReg) Loss for contrastive learning.

    Args:
        lambda_var (float): Weight for the variance term.
        mu_mean (float): Weight for the mean term.
        nu_cov (float): Weight for the covariance term.
    """
    def __init__(self, lambda_var=25, mu_mean=25, nu_cov=1):
        super(VicRegLoss, self).__init__()
        self.lambda_var = lambda_var
        self.mu_mean = mu_mean
        self.nu_cov = nu_cov

    def forward(self, z1, z2):
        """
        Compute the VicReg loss between two sets of embeddings.

        Args:
            z1 (torch.Tensor): First set of embeddings.
            z2 (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: Computed VicReg loss.
        """
        # Variance loss
        variance_loss = torch.mean(torch.relu(1 - torch.std(z1, dim=0))) + \
                        torch.mean(torch.relu(1 - torch.std(z2, dim=0)))

        # Mean loss
        mean_loss = torch.mean((torch.mean(z1, dim=0) - torch.mean(z2, dim=0))**2)

        # Covariance loss
        def compute_covariance_loss(z):
            z_centered = z - z.mean(dim=0)
            covariance_matrix = torch.mm(z_centered.T, z_centered) / (z.size(0) - 1)
            off_diagonal_sum = torch.sum(covariance_matrix ** 2) - torch.sum(torch.diag(covariance_matrix) ** 2)
            return off_diagonal_sum

        covariance_loss = compute_covariance_loss(z1) + compute_covariance_loss(z2)

        # Total loss
        total_loss = self.lambda_var * variance_loss + \
                     self.mu_mean * mean_loss + \
                     self.nu_cov * covariance_loss
        return total_loss

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss for contrastive learning.

    Args:
        temperature (float): Scaling factor for similarity scores.
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss between two sets of embeddings.

        Args:
            z_i (torch.Tensor): First set of embeddings.
            z_j (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: Computed NT-Xent loss.
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask for positives and negatives
        mask = ~torch.eye(2 * batch_size, device=z.device).bool()
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ])
        negatives = similarity_matrix.masked_select(mask).view(2 * batch_size, -1)

        # Compute NT-Xent loss
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(negatives), dim=-1)
        return -torch.mean(torch.log(numerator / denominator))

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.

    Args:
        margin (float): Margin for triplet loss.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1.0 - F.cosine_similarity(a, b),
            margin=self.margin
        )

    def forward(self, anchor, positive, negative):
        """
        Compute the Triplet loss.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.

        Returns:
            torch.Tensor: Computed Triplet loss.
        """
        return self.criterion(anchor, positive, negative)

# Contrastive Learning Loss Functions
def contrastive_loss(z1, z2, temperature=0.5):
    """
    Compute the basic contrastive loss using a similarity matrix and CrossEntropyLoss.

    Args:
        z1 (torch.Tensor): First set of embeddings.
        z2 (torch.Tensor): Second set of embeddings.
        temperature (float): Scaling factor for similarity scores.

    Returns:
        torch.Tensor: Computed contrastive loss.
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    similarity_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return nn.CrossEntropyLoss()(similarity_matrix, labels)


def info_nce_loss(z1, z2, temperature=0.5):
    """
    Compute the InfoNCE (Info Noise Contrastive Estimation) loss.

    Args:
        z1 (torch.Tensor): First set of embeddings.
        z2 (torch.Tensor): Second set of embeddings.
        temperature (float): Scaling factor for similarity scores.

    Returns:
        torch.Tensor: Computed InfoNCE loss.
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    batch_size = z1.size(0)
    similarity_matrix = torch.mm(z1, z2.T) / temperature

    pos_mask = torch.eye(batch_size, device=z1.device)
    neg_mask = 1 - pos_mask

    numerator = torch.exp(similarity_matrix * pos_mask)
    denominator = torch.sum(torch.exp(similarity_matrix * neg_mask), dim=1, keepdim=True) + numerator

    loss = -torch.log(numerator / denominator)
    return loss.mean()

class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, z_a, z_b, predictor):
        """
        Compute the BYOL loss between two sets of embeddings.

        Args:
            z_a (torch.Tensor): First set of embeddings.
            z_b (torch.Tensor): Second set of embeddings.
            predictor (nn.Module): Predictor network.

        Returns:
            torch.Tensor: Computed BYOL loss.
        """
        # Normalize embeddings
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        # Predict z_b from z_a
        p_a = predictor(z_a)
        p_a = F.normalize(p_a, dim=1)

        # Compute MSE loss between predicted and target embeddings
        loss = 2 - 2 * (p_a * z_b).sum(dim=1).mean()
        return loss

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=50):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive learning.

    Args:
        input_dim (int): Input dimension of embeddings.
        projection_dim (int): Output dimension of projected embeddings.
    """
    def __init__(self, input_dim, projection_dim=128):
        super(ContrastiveHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        """
        Forward pass for the projection head.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        return self.projector(x)

# Augmented NT-Xent Loss
def compute_nt_xent_loss_with_augmentation(images, model, contrastive_head, temperature=0.5):
    """
    Compute NT-Xent loss with data augmentation.

    Args:
        images (torch.Tensor): Input images.
        model (nn.Module): Base model for feature extraction.
        contrastive_head (ContrastiveHead): Projection head for embeddings.
        temperature (float): Scaling factor for similarity scores.

    Returns:
        torch.Tensor: Computed NT-Xent loss.
    """
    # Generate two augmented views
    augmented_1 = augment(images)
    augmented_2 = augment(images)

    # Forward pass through the model and projection head
    z1 = contrastive_head(model(augmented_1))
    z2 = contrastive_head(model(augmented_2))

    # Compute NT-Xent loss
    loss_fn = NTXentLoss(temperature=temperature)
    return loss_fn(z1, z2)

# Augmented Triplet Loss
def compute_triplet_loss_with_augmentation(images, model, contrastive_head, margin=1.0):
    """
    Compute Triplet loss with data augmentation.

    Args:
        images (torch.Tensor): Input images.
        model (nn.Module): Base model for feature extraction.
        contrastive_head (ContrastiveHead): Projection head for embeddings.
        margin (float): Margin for triplet loss.

    Returns:
        torch.Tensor: Computed Triplet loss.
    """
    # Generate anchor, positive, and negative samples
    indices = torch.randperm(images.size(0))
    anchor_images = images
    positive_images = images[indices]
    negative_images = images[torch.randperm(images.size(0))]

    # Augment images
    anchor_images = augment(anchor_images)
    positive_images = augment(positive_images)
    negative_images = augment(negative_images)

    # Forward pass through the model and projection head
    anchor_embeddings = contrastive_head(model(anchor_images))
    positive_embeddings = contrastive_head(model(positive_images))
    negative_embeddings = contrastive_head(model(negative_images))

    # Compute Triplet loss
    loss_fn = TripletLoss(margin=margin)
    return loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
