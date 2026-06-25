import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_lesion_images():
    # 1. Create sharp skin lesion simulation
    # Image size 400x400
    base_skin = Image.new("RGB", (400, 400), (245, 220, 195)) # skin tone
    draw = ImageDraw.Draw(base_skin)
    
    # Draw a brown, slightly asymmetrical skin mole in the center
    draw.ellipse([140, 130, 260, 270], fill=(95, 55, 30))
    draw.ellipse([150, 150, 240, 250], fill=(70, 38, 18))
    # Add a tiny dark spot
    draw.ellipse([170, 180, 190, 200], fill=(30, 15, 8))
    
    # Apply a light filter to blend edges
    blended = base_skin.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Convert to numpy to add high-frequency skin pore noise
    skin_np = np.array(blended).astype(np.float32)
    
    # Generate Gaussian noise (pores and fine texture)
    # Mean = 0, Std Dev = 12 (adds subtle texture)
    noise = np.random.normal(0, 12, skin_np.shape).astype(np.float32)
    textured_skin = np.clip(skin_np + noise, 0, 255).astype(np.uint8)
    
    # Save textured sharp image
    sharp_img = Image.fromarray(textured_skin)
    sharp_img.save("clear_lesion.jpg", "JPEG", quality=95)
    print("Created sharp lesion test image with skin pores: clear_lesion.jpg")

    # 2. Create blurry skin lesion simulation
    # Apply a heavy Gaussian blur to the textured image to fail the Laplacian check
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=6.0))
    blurry_img.save("blurry_lesion.jpg", "JPEG", quality=95)
    print("Created blurry lesion test image: blurry_lesion.jpg")

if __name__ == "__main__":
    create_lesion_images()
