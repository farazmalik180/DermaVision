import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO

def check_image_blur(image_bytes, threshold=60.0):
    """
    Computes the Laplacian variance of the image to check if it's too blurry.
    A variance value below `threshold` indicates a blurry image.
    """
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return True, 0.0, "Failed to decode image data"
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = lap_var < threshold
        return is_blurry, lap_var, None
    except Exception as e:
        return True, 0.0, f"Error calculating image blur: {str(e)}"

def preprocess_image(image_bytes):
    """
    Loads an image from bytes, resizes to 224x224, normalizes with ImageNet parameters,
    and returns a preprocessed PyTorch tensor and the original image in [0, 1] RGB numpy float32.
    """
    # Open PIL Image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Keep original image as float32 in [0, 1] for Grad-CAM
    original_np = np.array(pil_image).astype(np.float32) / 255.0
    
    # Resize to 224x224
    resized_pil = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
    
    # ImageNet Mean & Std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized_np = (resized_np - mean) / std
    
    # Convert from HWC to CHW and wrap in a batch tensor [1, C, H, W]
    tensor = torch.from_numpy(normalized_np).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, original_np
