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

def is_dermoscopy_image(image_bytes, non_skin_threshold=0.6):
    """
    Checks if the image is likely a dermoscopy photograph by analyzing color histograms.
    Rejects images with too many non-skin colors (blue, bright green, heavy white background).
    Returns (is_valid, reason)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Failed to decode image data"
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Skin color bounds in HSV (OpenCV uses H: 0-179, S: 0-255, V: 0-255)
        # Skin is usually H: 0-20 or H: 160-179 (Red/Pink/Brown)
        # Also need to exclude pure white/black (low S, very high/low V)
        
        # Define ranges for "non-skin" colors (e.g. blues, greens)
        # Hue 30 to 150 covers green to blue to purple
        lower_non_skin = np.array([30, 40, 40])
        upper_non_skin = np.array([150, 255, 255])
        
        non_skin_mask = cv2.inRange(hsv, lower_non_skin, upper_non_skin)
        
        # Define range for pure white/gray backgrounds (often seen in non-medical photos)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 30, 255])
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined_invalid_mask = cv2.bitwise_or(non_skin_mask, white_mask)
        
        # Calculate percentage of invalid pixels
        invalid_ratio = np.sum(combined_invalid_mask > 0) / (img.shape[0] * img.shape[1])
        
        if invalid_ratio > non_skin_threshold:
            return False, f"Image contains {invalid_ratio*100:.1f}% non-skin colors. Please upload a close-up dermoscopy photo."
            
        return True, ""
    except Exception as e:
        return False, f"Color analysis failed: {str(e)}"

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
