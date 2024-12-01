# A program to apply the top k numbers of eigenvectors to images with the purpose
# to reconstruct the images with reduced file size and pixel dimensions. This approach
# may benefit the application of machine vision in a low-memory and low-processing devices
# such as raspberry Pi, Arduino board, or NVIDIA Jetson Nano.

# Authors: Yusnaidi Md Yusof
# Date: 2 December 2024
# Email: yusnaidi.kl@utm.my


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, downscale_factor=4):
    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)  # Convert to grayscale
    reduced_image = grayscale_image[::downscale_factor, ::downscale_factor]  # Downscale
    return reduced_image

# Interactive menu for user input
def interactive_menu():
    print("=== Image Reconstruction with PCA ===")
    image_path = input("Enter the path to the image file: ").strip()
    while not os.path.exists(image_path):
        print("Invalid file path. Please try again.")
        image_path = input("Enter the path to the image file: ").strip()

    try:
        downscale_factor = int(input("Enter the downscale factor (default=4): ").strip() or 4)
    except ValueError:
        print("Invalid input. Using default downscale factor: 4")
        downscale_factor = 4

    try:
        top_k = int(input("Enter the number of top eigenvectors to use (default=10): ").strip() or 10)
    except ValueError:
        print("Invalid input. Using default top_k: 10")
        top_k = 10

    return image_path, downscale_factor, top_k

# Main function
if __name__ == "__main__":
    # Interactive menu
    image_path, downscale_factor, top_k = interactive_menu()

    # Step 1: Load and preprocess the image
    image = load_and_preprocess_image(image_path, downscale_factor)
    original_shape = image.shape

    # Display the preprocessed image
    plt.figure(figsize=(6, 6))
    plt.title("Preprocessed Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

    # Step 2: Flatten the image and center the data
    X = image.reshape(-1, original_shape[1])  # Flatten rows
    X_mean = np.mean(X, axis=0)  # Compute mean of each column
    X_centered = X - X_mean  # Center the data

    # Step 3: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Step 4: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 5: Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Descending order
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Display top k eigenvectors' information
    total_variance = np.sum(sorted_eigenvalues)
    top_k_variance = sorted_eigenvalues[:top_k]
    top_k_vectors = sorted_eigenvectors[:, :top_k]
    explained_variance_ratio = (top_k_variance / total_variance) * 100

    print("\nTop k Eigenvectors and Variance Explained:")
    for i, (eig_val, var_ratio, eig_vec) in enumerate(zip(top_k_variance, explained_variance_ratio, top_k_vectors.T)):
        print(f"Eigenvalue {i + 1}: {eig_val:.4f}")
        print(f"Explained Variance: {var_ratio:.2f}%")
        print(f"Eigenvector {i + 1}:\n{eig_vec}\n")

    # Step 7: Reconstruct the image using top k eigenvectors
    X_projected = X_centered @ top_k_vectors  # Project data onto top k eigenvectors
    X_reconstructed = (X_projected @ top_k_vectors.T) + X_mean  # Reconstruct from projection

    # Reshape back to the original dimensions
    image_reconstructed = X_reconstructed.reshape(original_shape)

    # Save the reconstructed image
    reconstructed_image_path = os.path.join(
        os.path.dirname(image_path), f"reconstructed_{os.path.basename(image_path)}"
    )
    io.imsave(reconstructed_image_path, (image_reconstructed * 255).astype(np.uint8))  # Scale to 0-255

    # Display the original and reconstructed images side by side
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed Image (Top {top_k} Eigenvectors)")
    plt.imshow(image_reconstructed, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Step 8: Analyze the reconstructed image
    original_pixel_count = original_shape[0] * original_shape[1]  # Total pixels in original image
    reconstructed_pixel_count = top_k * original_shape[1]  # Total pixels from k eigenvectors

    size_reduction_percentage = 100 - ((reconstructed_pixel_count / original_pixel_count) * 100)

    print("\n--- Analysis of Reconstructed Image ---")
    print(f"Original Image Size (pixels): {original_pixel_count}")
    print(f"Reconstructed Image Size (pixels): {reconstructed_pixel_count}")
    print(f"Number of Eigenvectors Used: {top_k}")
    print(f"Image Size Reduction: {size_reduction_percentage:.2f}%")
    print(f"Reconstructed Image Saved at: {os.path.abspath(reconstructed_image_path)}")