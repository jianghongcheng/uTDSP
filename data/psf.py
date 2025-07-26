import torch
import torch.nn.functional as F
import numpy as np

def gaussian_blur(input_tensor, kernel_size=15, sigma=3.4):
    # Create 2D Gaussian kernel
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel_1d = (x / x.sum()).unsqueeze(0)
    
    # Create 2D Gaussian kernel by multiplying two 1D kernels
    kernel_2d = kernel_1d.T @ kernel_1d
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size, kernel_size)
    
    # Apply Gaussian blur to each channel (band) using 2D convolution
    blurred_tensor = input_tensor.clone()
    for i in range(input_tensor.shape[1]):  # Loop over bands
        blurred_tensor[:, i:i+1, :, :] = F.conv2d(input_tensor[:, i:i+1, :, :], kernel_2d, padding=kernel_size//2)
    
    return blurred_tensor

def downsample(input_tensor, scale_factor=8):
    # Downsample the image by a factor of 8
    return F.interpolate(input_tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)

def add_gaussian_noise(input_tensor, snr_db=20):
    # Calculate the signal power
    signal_power = torch.mean(input_tensor**2)
    
    # Calculate noise power based on desired SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = torch.sqrt(noise_power) * torch.randn_like(input_tensor)
    
    # Add noise to the input tensor
    noisy_tensor = input_tensor + noise
    
    return noisy_tensor

# Example function to achieve the PSF B
def apply_psf(input_tensor):
    # Step 1: Apply Gaussian blur to each band
    blurred = gaussian_blur(input_tensor)
    
    # Step 2: Downsample the blurred image
    downsampled = downsample(blurred)
    
    # Step 3: Add Gaussian noise (20 dB SNR)
    noisy_image = add_gaussian_noise(downsampled)
    
    return noisy_image

# Example usage:
# Assuming input_tensor is a hyperspectral image of shape (batch_size, num_bands, height, width)
input_tensor = torch.randn(1, 191, 256, 256)  # Example tensor
output = apply_psf(input_tensor)

print(output.shape)  # Resulting LR-HSI shape after downsampling and noise
