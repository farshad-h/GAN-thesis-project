# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

def vae_loss(recon_x, x, mu, logvar, beta=1):
    """
    Compute the VAE loss with reconstruction and KL divergence terms.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean from the latent space.
        logvar (torch.Tensor): Log variance from the latent space.
        beta (float): Weight for the KL divergence term.

    Returns:
        torch.Tensor: Total loss (reconstruction + beta * KL divergence).
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    return recon_loss + beta * kld_loss


def vae_ssim_loss(recon_x, x, mu, logvar, beta=1, ssim_func=None):
    """
    Compute the VAE loss with SSIM-based reconstruction and KL divergence terms.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean from the latent space.
        logvar (torch.Tensor): Log variance from the latent space.
        beta (float): Weight for the KL divergence term.
        ssim_func (callable): SSIM function for image similarity.

    Returns:
        torch.Tensor: Total loss (SSIM + beta * KL divergence).

    Raises:
        ValueError: If `ssim_func` is not provided.
    """
    if ssim_func is None:
        raise ValueError("A valid SSIM function must be provided for vae_ssim_loss.")

    # Reconstruction loss (SSIM)
    recon_loss = 1 - ssim_func(recon_x, x)

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    return recon_loss + beta * kld_loss

def loss_function_dae_ssim(recon_x, x, ssim_func):
    """
    Compute SSIM-based loss for denoising autoencoder.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        ssim_func (callable): SSIM function.

    Returns:
        torch.Tensor: SSIM-based reconstruction loss.
    """
    return 1 - ssim_func(recon_x, x)

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


def linear_beta_schedule(epoch, num_epochs, initial_beta=0.0, final_beta=1.0):
    """
    Compute the beta value for linear beta scheduling.

    Args:
        epoch (int): Current epoch.
        num_epochs (int): Total number of epochs.
        initial_beta (float): Initial value of beta.
        final_beta (float): Final value of beta.

    Returns:
        float: Linearly interpolated beta value.
    """
    return initial_beta + (final_beta - initial_beta) * (epoch / num_epochs)

def cyclical_beta_schedule(epoch, cycle_length=10):
    """
    Compute the beta value for cyclical beta scheduling.

    Args:
        epoch (int): Current epoch.
        cycle_length (int): Length of each cycle (in epochs).

    Returns:
        float: Cyclically interpolated beta value.
    """
    return (epoch % cycle_length) / cycle_length