from PIL import Image

# Load image
img = Image.open("VanGogh.jpg")

# Resize to 512x512
img_resized = img.resize((512, 512), Image.LANCZOS)

# Save
img_resized.save("VanGogh.jpg")

print("Saved resized image:", img_resized.size)