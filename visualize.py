import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image


LATENT_SCALE = 0.18215


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Stable Diffusion VAE latents with t-SNE and per-channel maps."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        default="latent_visualizations",
        help="Directory where visualizations will be saved.",
    )
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model id or local path that contains a VAE subfolder.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for VAE encoding.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Torch dtype. auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=768,
        help="Resize image so total pixels are at most max_resolution squared.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=5000,
        help="Maximum number of latent spatial positions used for t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It is clipped automatically for small samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for t-SNE sampling.",
    )
    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Also decode the latent through the VAE and save the reconstruction.",
    )
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device_arg


def choose_dtype(dtype_arg, device):
    if dtype_arg == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    return torch.float32


def resize_to_sd_grid(image, max_resolution):
    width_original, height_original = image.size
    scale = (max_resolution * max_resolution / (width_original * height_original)) ** 0.5
    if scale > 1:
        scale = 1

    width = max(64, int(round(width_original * scale / 64)) * 64)
    height = max(64, int(round(height_original * scale / 64)) * 64)
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.LANCZOS)


def image_to_sd_tensor(image, device, dtype):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) * 2.0
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def encode_image(vae, image_tensor):
    posterior = vae.encode(image_tensor).latent_dist
    latent = posterior.mean * LATENT_SCALE
    return latent


@torch.no_grad()
def decode_latent(vae, latent):
    decoded = vae.decode(latent / LATENT_SCALE).sample
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
    image = decoded[0].detach().float().cpu().permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


def minmax_normalize(array):
    array_min = np.min(array)
    array_max = np.max(array)
    if np.isclose(array_min, array_max):
        return np.zeros_like(array)
    return (array - array_min) / (array_max - array_min)


def save_channel_maps(latent, output_dir, stem):
    latent_np = latent[0].detach().float().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for channel_idx, axis in enumerate(axes.flat):
        channel = latent_np[channel_idx]
        im = axis.imshow(channel, cmap="viridis")
        axis.set_title(f"Latent channel {channel_idx}")
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Stable Diffusion VAE latent channels")
    path = output_dir / f"{stem}_latent_channels.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    for channel_idx in range(latent_np.shape[0]):
        channel = minmax_normalize(latent_np[channel_idx])
        plt.imsave(output_dir / f"{stem}_latent_channel_{channel_idx}.png", channel, cmap="viridis")


def load_tsne():
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "t-SNE visualization requires scikit-learn. Install it with: pip install scikit-learn"
        ) from exc
    return TSNE


def save_tsne(latent, output_dir, stem, sample_points, perplexity, seed):
    TSNE = load_tsne()

    latent_np = latent[0].detach().float().cpu().numpy()
    channels, height, width = latent_np.shape
    vectors = latent_np.reshape(channels, height * width).T

    rng = np.random.default_rng(seed)
    total_points = vectors.shape[0]
    if sample_points > 0 and total_points > sample_points:
        indices = rng.choice(total_points, size=sample_points, replace=False)
        vectors_for_tsne = vectors[indices]
    else:
        indices = np.arange(total_points)
        vectors_for_tsne = vectors

    point_count = vectors_for_tsne.shape[0]
    if point_count < 3:
        raise ValueError("Need at least 3 latent positions for t-SNE.")

    perplexity = min(perplexity, max(1.0, (point_count - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    embedding = tsne.fit_transform(vectors_for_tsne)

    rows = indices // width
    cols = indices % width
    spatial_colors = np.stack(
        [
            cols / max(width - 1, 1),
            rows / max(height - 1, 1),
            minmax_normalize(np.linalg.norm(vectors_for_tsne, axis=1)),
        ],
        axis=1,
    )

    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.scatter(embedding[:, 0], embedding[:, 1], c=spatial_colors, s=5, alpha=0.85, linewidths=0)
    axis.set_title(f"t-SNE of {point_count} latent vectors")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    fig.savefig(output_dir / f"{stem}_latent_tsne_scatter.png", dpi=220)
    plt.close(fig)

    tsne_image = np.zeros((height * width, 3), dtype=np.float32)
    tsne_xy = minmax_normalize(embedding)
    tsne_image[indices, 0] = tsne_xy[:, 0]
    tsne_image[indices, 1] = tsne_xy[:, 1]
    tsne_image[indices, 2] = 1.0
    tsne_image = tsne_image.reshape(height, width, 3)
    tsne_image = F.interpolate(
        torch.from_numpy(tsne_image).permute(2, 0, 1).unsqueeze(0),
        scale_factor=8,
        mode="nearest",
    )[0].permute(1, 2, 0).numpy()
    plt.imsave(output_dir / f"{stem}_latent_tsne_grid.png", tsne_image)


def save_latent_tensor(latent, output_dir, stem):
    np.save(output_dir / f"{stem}_latent.npy", latent[0].detach().float().cpu().numpy())


def main():
    args = parse_args()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    image = resize_to_sd_grid(image, args.max_resolution)
    image.save(output_dir / f"{image_path.stem}_input_resized.png")

    image_tensor = image_to_sd_tensor(image, device=device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()

    latent = encode_image(vae, image_tensor)
    stem = image_path.stem
    save_latent_tensor(latent, output_dir, stem)
    save_channel_maps(latent, output_dir, stem)
    save_tsne(
        latent=latent,
        output_dir=output_dir,
        stem=stem,
        sample_points=args.sample_points,
        perplexity=args.perplexity,
        seed=args.seed,
    )

    if args.save_reconstruction:
        reconstruction = decode_latent(vae, latent)
        Image.fromarray(reconstruction).save(output_dir / f"{stem}_vae_reconstruction.png")

    print(f"Saved latent visualizations to: {output_dir.resolve()}")
    print(f"Latent shape: {tuple(latent.shape)}")


if __name__ == "__main__":
    main()
