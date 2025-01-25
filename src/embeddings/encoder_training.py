# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from typing import Callable, Optional
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

class EarlyStopping:
    """
    Early stopping to stop training when the loss does not improve after a specified number of epochs (patience).
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, current_loss):
        """
        Check if training should stop.

        Args:
            current_loss (float): Current epoch's loss.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

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

import inspect

def train_autoencoder_v4(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    epochs: int = 10,
    device: str = "cpu",
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    contrastive_loss_fn: Optional[Callable] = None,
    temperature: float = 0.5,
    triplet_data: bool = False,
    augment_fn: Optional[Callable] = None,
    predictor: Optional[nn.Module] = None,  # Add predictor for BYOL
    patience: int = 5,
    min_delta: float = 0.001,
):
    """
    Unified training function for autoencoders with support for:
    - Reconstruction loss
    - Contrastive loss (e.g., NT-Xent, VicReg, Triplet, Contrastive, InfoNCE, Barlow Twins, BYOL)
    - Noise injection (for denoising autoencoders)
    - Data augmentation
    - Early stopping
    """
    model.to(device).train()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            # Prepare data based on whether it's triplet data or not
            if triplet_data:
                anchor, positive, negative = batch
                anchor, positive, negative = (
                    anchor.to(device).float(),
                    positive.to(device).float(),
                    negative.to(device).float(),
                )
                images = anchor  # Use anchor as the primary input for reconstruction
            else:
                images, _ = batch
                images = images.to(device).float()
            
            encoded, decoded = model(images)

            # Compute reconstruction loss
            reconstruction_loss = loss_fn(decoded, images)

            # Compute contrastive loss if specified
            contrastive_loss_value = 0
            if contrastive_loss_fn is not None:
                if triplet_data:
                    # Triplet loss
                    positive_encoded, _ = model(positive)
                    negative_encoded, _ = model(negative)

                    # Flatten embeddings
                    encoded = encoded.view(encoded.size(0), -1)
                    positive_encoded = positive_encoded.view(positive_encoded.size(0), -1)
                    negative_encoded = negative_encoded.view(negative_encoded.size(0), -1)

                    # Compute triplet loss
                    contrastive_loss_value = contrastive_loss_fn(encoded, positive_encoded, negative_encoded)
                else:
                    # NT-Xent, VicReg, Contrastive, InfoNCE, Barlow Twins, BYOL, or other contrastive loss
                    if augment_fn:
                        augmented_1 = augment_fn(images)
                        augmented_2 = augment_fn(images)
                        z1, _ = model(augmented_1)
                        z2, _ = model(augmented_2)
                    else:
                        z1, z2 = encoded, encoded  # Use the same embeddings if no augmentation

                    # Flatten embeddings
                    z1 = z1.view(z1.size(0), -1)
                    z2 = z2.view(z2.size(0), -1)

                    # Handle all contrastive losses uniformly
                    if isinstance(contrastive_loss_fn, BYOLLoss):
                        # BYOL requires a predictor
                        if predictor is None:
                            raise ValueError("Predictor network must be provided for BYOL loss.")
                        contrastive_loss_value = contrastive_loss_fn(z1, z2, predictor=predictor)
                    else:
                        # # Check if the loss function accepts a `temperature` parameter
                        # if "temperature" in inspect.signature(contrastive_loss_fn.forward).parameters:
                        #     contrastive_loss_value = contrastive_loss_fn(z1, z2, temperature=temperature)
                        # else:
                        #     contrastive_loss_value = contrastive_loss_fn(z1, z2)
                        # Check if the loss function accepts a `temperature` parameter
                        if "temperature" in inspect.signature(contrastive_loss_fn).parameters:
                            contrastive_loss_value = contrastive_loss_fn(z1, z2, temperature=temperature)
                        else:
                            contrastive_loss_value = contrastive_loss_fn(z1, z2)

            # Total loss
            total_loss_value = reconstruction_loss + contrastive_loss_value

            # Backpropagation
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            total_loss += total_loss_value.item()

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

        # Compute average epoch loss
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}")

        # Check for early stopping
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

def train_autoencoder(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    epochs: int = 10,
    device: str = "cpu",
    noise_factor: float = 0.0,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    contrastive_loss_fn: Optional[Callable] = None,
    temperature: float = 0.5,
    triplet_data: bool = False,
    augment_fn: Optional[Callable] = None,
    patience: int = 5,
    min_delta: float = 0.001,
):
    """
    Unified training function for autoencoders with support for:
    - Reconstruction loss
    - Contrastive loss (e.g., NT-Xent, InfoNCE)
    - Triplet loss
    - Noise injection (for denoising autoencoders)
    - Data augmentation
    - Early stopping

    Args:
        model (nn.Module): The autoencoder model.
        data_loader (DataLoader): DataLoader for training data.
        loss_fn (Callable): Primary loss function (e.g., reconstruction loss).
        optimizer (optim.Optimizer): Optimizer for the model.
        epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu' or 'cuda').
        noise_factor (float): Factor for adding noise to input images (denoising autoencoder).
        scheduler (Optional[optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        contrastive_loss_fn (Optional[Callable]): Contrastive loss function (e.g., NT-Xent, triplet loss).
        temperature (float): Temperature parameter for NT-Xent loss.
        triplet_data (bool): Whether the data_loader provides triplets (anchor, positive, negative).
        augment_fn (Optional[Callable]): Augmentation function for contrastive learning.
        patience (int): Number of epochs with no significant improvement before triggering early stopping.
        min_delta (float): Minimum change in loss to qualify as an improvement.

    Returns:
        None: Prints loss values for each epoch.
    """
    model.to(device).train()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            # Prepare data based on whether it's triplet data or not
            if triplet_data:
                anchor, positive, negative = batch
                anchor, positive, negative = (
                    anchor.to(device).float(),
                    positive.to(device).float(),
                    negative.to(device).float(),
                )
                images = anchor  # Use anchor as the primary input for reconstruction
            else:
                images, _ = batch
                images = images.to(device).float()

            # Add noise if specified
            if noise_factor > 0:
                noisy_images = images + noise_factor * torch.randn_like(images)
                noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
                encoded, decoded = model(noisy_images)
            else:
                encoded, decoded = model(images)

            # Compute reconstruction loss
            reconstruction_loss = loss_fn(decoded, images)

            # Compute contrastive loss if specified
            contrastive_loss_value = 0
            if contrastive_loss_fn is not None:
                if triplet_data:
                    # Triplet loss
                    positive_encoded, _ = model(positive)
                    negative_encoded, _ = model(negative)
                    contrastive_loss_value = contrastive_loss_fn(encoded, positive_encoded, negative_encoded)
                else:
                    # NT-Xent or other contrastive loss
                    if augment_fn:
                        augmented_1 = augment_fn(images)
                        augmented_2 = augment_fn(images)
                        z1, _ = model(augmented_1)
                        z2, _ = model(augmented_2)
                    else:
                        z1, z2 = encoded, encoded  # Use the same embeddings if no augmentation
                    contrastive_loss_value = contrastive_loss_fn(z1, z2, temperature)

            # Total loss
            total_loss_value = reconstruction_loss + contrastive_loss_value

            # Backpropagation
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            total_loss += total_loss_value.item()

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

        # Compute average epoch loss
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}")

        # Check for early stopping
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

def train_vae(
    vae: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int = 10,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_best: bool = False,
    save_path: str = "best_vae_model.pth",
    beta: float = 1.0,
    alpha: Optional[float] = None,
    temperature: float = 0.5,
    contrastive_loss_fn: Optional[Callable] = None,
    patience: int = 5,
    min_delta: float = 0.001,
):
    """
    Unified training function for VAEs with support for:
    - Reconstruction loss (e.g., MSE, SSIM).
    - KL divergence.
    - Contrastive learning (e.g., NT-Xent).
    - Optional validation and model saving.
    - Early stopping

    Args:
        vae (nn.Module): VAE model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (Callable): Loss function for reconstruction and KL divergence.
        epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu' or 'cuda').
        val_loader (Optional[DataLoader]): DataLoader for validation data.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        save_best (bool): Whether to save the best model based on validation loss.
        save_path (str): Path to save the best model.
        beta (float): Weight for the KL divergence term.
        alpha (Optional[float]): Weight for the contrastive loss term.
        temperature (float): Temperature parameter for contrastive loss.
        contrastive_loss_fn (Optional[Callable]): Contrastive loss function (e.g., NT-Xent).
        patience (int): Number of epochs with no significant improvement before triggering early stopping.
        min_delta (float): Minimum change in loss to qualify as an improvement.

    Returns:
        None: Prints loss values for each epoch.
    """
    vae.to(device).train()
    best_val_loss = float('inf') if save_best else None

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        # Training loop
        vae.train()
        total_train_loss = 0
        for images, _ in train_loader:
            images = images.to(device).float()

            # Forward pass
            mu, logvar, decoded = vae(images)

            # Compute reconstruction and KL divergence loss
            recon_loss = loss_fn(decoded, images, mu, logvar, beta)

            # Compute contrastive loss if specified
            contrastive_loss_value = 0
            if contrastive_loss_fn is not None and alpha is not None:
                if hasattr(vae, 'projection_head'):
                    indices = torch.randperm(mu.size(0)).to(device)
                    z1, z2 = mu, mu[indices]
                    contrastive_loss_value = contrastive_loss_fn(z1, z2, temperature)
                else:
                    raise ValueError("VAE model must have a projection head for contrastive loss.")

            # Total loss
            total_loss = recon_loss + (alpha * contrastive_loss_value if alpha is not None else 0)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        # Validation loop
        if val_loader:
            vae.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device).float()
                    mu, logvar, decoded = vae(images)
                    val_loss = loss_fn(decoded, images, mu, logvar, beta)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

            # Save the best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(vae.state_dict(), save_path)
                print(f"Model saved at epoch {epoch + 1}")

            # Check for early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        else:
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

            # Check for early stopping (using training loss if no validation data)
            early_stopping(avg_train_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

def add_noise(inputs, noise_factor=0.1):
    """
    Add Gaussian noise to the input tensor.

    Args:
        inputs (torch.Tensor): Input tensor.
        noise_factor (float): Scaling factor for noise.

    Returns:
        torch.Tensor: Noisy tensor with values clamped between 0 and 1.
    """
    noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy_inputs, 0., 1.)

def train_dae(
    dae: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int = 10,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_best: bool = False,
    save_path: str = "best_dae_model.pth",
    noise_factor: float = 0.1,
    alpha: Optional[float] = None,
    temperature: float = 0.5,
    contrastive_loss_fn: Optional[Callable] = None,
    triplet_loss_fn: Optional[Callable] = None,
    ssim_func: Optional[Callable] = None,
    patience: int = 5,
    min_delta: float = 0.001,
):
    """
    Unified training function for denoising autoencoders with support for:
    - Reconstruction loss (e.g., MSE, SSIM).
    - Contrastive learning (e.g., NT-Xent).
    - Triplet loss.
    - Noise injection.
    - Optional validation and model saving.
    - Early Stopping
    Args:
        dae (nn.Module): Denoising autoencoder model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (Callable): Loss function for reconstruction.
        epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu' or 'cuda').
        val_loader (Optional[DataLoader]): DataLoader for validation data.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        save_best (bool): Whether to save the best model based on validation loss.
        save_path (str): Path to save the best model.
        noise_factor (float): Noise scaling factor.
        alpha (Optional[float]): Weight for the contrastive or triplet loss term.
        temperature (float): Temperature parameter for contrastive loss.
        contrastive_loss_fn (Optional[Callable]): Contrastive loss function (e.g., NT-Xent).
        triplet_loss_fn (Optional[Callable]): Triplet loss function.
        ssim_func (Optional[Callable]): SSIM function for SSIM-based reconstruction loss.
        patience (int): Number of epochs with no significant improvement before triggering early stopping.
        min_delta (float): Minimum change in loss to qualify as an improvement.

    Returns:
        None: Prints loss values for each epoch.
    """
    dae.to(device).train()
    best_val_loss = float('inf') if save_best else None

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(epochs):
        # Training loop
        dae.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            # Add noise to the input images
            noisy_images = add_noise(images, noise_factor)

            # Forward pass
            outputs = dae(noisy_images)
            if len(outputs) == 3:
                projected_encoded, encoded, decoded = outputs
            else:
                encoded, decoded = outputs

            # Compute reconstruction loss
            if ssim_func:
                recon_loss = 1 - ssim_func(decoded, images)  # SSIM-based reconstruction loss
            else:
                recon_loss = loss_fn(decoded, images)  # Standard reconstruction loss (e.g., MSE)

            # Compute contrastive loss if specified
            contrastive_loss_value = 0
            if contrastive_loss_fn is not None and alpha is not None:
                indices = torch.randperm(len(encoded)).to(device)
                z1, z2 = encoded, encoded[indices]
                contrastive_loss_value = contrastive_loss_fn(z1, z2, temperature)

            # Compute triplet loss if specified
            triplet_loss_value = 0
            if triplet_loss_fn is not None and alpha is not None:
                triplets = []
                for i in range(len(labels)):
                    anchor_label = labels[i]
                    positive_indices = torch.where(labels == anchor_label)[0]
                    if len(positive_indices) > 1:
                        positive_index = positive_indices[positive_indices != i][torch.randint(len(positive_indices) - 1, (1,))].item()
                        negative_label = labels[labels != anchor_label][torch.randint(len(labels[labels != anchor_label]), (1,))].item()
                        negative_index = torch.where(labels == negative_label)[0][torch.randint(len(labels[labels == negative_label]), (1,))].item()
                        triplets.append((encoded[i], encoded[positive_index], encoded[negative_index]))

                if triplets:
                    anchor_embeddings, positive_embeddings, negative_embeddings = zip(*triplets)
                    anchor_embeddings = torch.stack(anchor_embeddings)
                    positive_embeddings = torch.stack(positive_embeddings)
                    negative_embeddings = torch.stack(negative_embeddings)
                    triplet_loss_value = triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Total loss
            total_loss = recon_loss + (alpha * contrastive_loss_value if contrastive_loss_fn else 0) + (alpha * triplet_loss_value if triplet_loss_fn else 0)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        # Validation loop
        if val_loader:
            dae.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device).float()
                    noisy_images = add_noise(images, noise_factor)
                    encoded, decoded = dae(noisy_images)
                    val_loss = loss_fn(decoded, images)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

            # Save the best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(dae.state_dict(), save_path)
                print(f"Model saved at epoch {epoch + 1}")

            # Check for early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        else:
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

            # Check for early stopping (using training loss if no validation data)
            early_stopping(avg_train_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

def train_simclr(model, dataloader, num_epochs=10, lr=1e-3, device=None):
    """
    Trains a SimCLR model.

    Args:
        model (SimCLR): SimCLR model to train.
        dataloader (DataLoader): DataLoader providing pairs of augmented images.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (torch.device, optional): Device to use (e.g., 'cuda' or 'cpu'). Defaults to None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x1, x2 in progress_bar:  # x1 and x2 are augmented views of the same image
            x1, x2 = x1.to(device), x2.to(device)
            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")