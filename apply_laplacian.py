import cv2
import matplotlib.pyplot as plt
import os

# Folder containing latent channel images
img_dir = "latent_visualizations"

# Latent channel image files
image_files = [
    "image_latent_channel_0.png",
    "image_latent_channel_1.png",
    "image_latent_channel_2.png",
    "image_latent_channel_3.png",
]

# Create figure
fig, axes = plt.subplots(4, 2, figsize=(12, 20))

for row, file_name in enumerate(image_files):

    img_path = os.path.join(img_dir, file_name)

    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur (recommended before Laplacian)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Normalize for visualization
    laplacian_vis = cv2.normalize(
        laplacian,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    # Plot original latent channel
    axes[row, 0].imshow(img, cmap="gray")
    axes[row, 0].set_title(f"{file_name}")
    axes[row, 0].axis("off")

    # Plot Laplacian
    axes[row, 1].imshow(laplacian_vis, cmap="gray")
    axes[row, 1].set_title(f"Laplacian of Channel {row}")
    axes[row, 1].axis("off")

# Layout
plt.tight_layout()

# Save figure
plt.savefig("latent_laplacian_visualization.png", dpi=300, bbox_inches="tight")

# Show
plt.show()