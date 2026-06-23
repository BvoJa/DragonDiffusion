from PIL import Image

# Load image
img = Image.open("horse_mask.png")

# Resize to 512x512
img_resized = img.resize((512, 512), Image.LANCZOS)

# Save
img_resized.save("horse_mask_512.png")

print("Saved resized image:", img_resized.size)