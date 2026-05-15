from PIL import Image

# Load image
img = Image.open("2.png")

# Resize to 512x512
img_resized = img.resize((512, 512), Image.LANCZOS)

# Save
img_resized.save("2-512.png")

print("Saved resized image:", img_resized.size)