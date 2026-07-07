import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Define skin lesion classes
CLASSES = [
    {"name": "Melanoma", "risk_level": "High", "desc": "Malignant skin cancer starting in melanocytes."},
    {"name": "Melanocytic Nevus", "risk_level": "Low", "desc": "Common benign mole."},
    {"name": "Basal Cell Carcinoma", "risk_level": "High", "desc": "Common slow-growing non-melanoma skin cancer."},
    {"name": "Actinic Keratosis", "risk_level": "Moderate", "desc": "Pre-cancerous dry, scaly patch of skin."},
    {"name": "Seborrheic Keratosis", "risk_level": "Low", "desc": "Common non-cancerous benign skin growth."},
    {"name": "Dermatofibroma", "risk_level": "Low", "desc": "Common benign firm red-to-brown nodule."},
    {"name": "Vascular Lesion", "risk_level": "Low", "desc": "Benign growths/marks made of blood vessels."}
]

class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Try loading with default ImageNet pre-trained weights, fallback to None if offline
        try:
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        except Exception:
            self.model = models.efficientnet_v2_s(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(weights_path=None):
    model = SkinLesionModel(num_classes=len(CLASSES))
    
    # If weights path is provided and exists, load it
    if weights_path and os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading local weights from {weights_path}: {e}. Running with base model.")
    else:
        # Save a mock model check point if weights don't exist, for testing purposes
        # (This avoids failing if the user wants to test checkpoints)
        print("No custom checkpoint loaded. Using default/initialized weights.")
        
    model.eval()
    return model

def generate_gradcam(model, input_tensor, original_img_np, target_category_idx):
    """
    Generates a Grad-CAM overlay on the original image.
    
    Parameters:
    - model: The PyTorch model
    - input_tensor: Preprocessed image tensor [1, 3, 224, 224]
    - original_img_np: Original image numpy array [H, W, 3], scale [0, 1]
    - target_category_idx: Index of the target class
    
    Returns:
    - base64 encoded string of the Grad-CAM visualization
    """
    try:
        # Target the last feature layer of EfficientNetV2
        target_layers = [model.model.features[-1]]
        
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Specify target category
        targets = [ClassifierOutputTarget(target_category_idx)]
        
        # Generate raw CAM heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Resize CAM to match original image dimensions
        h, w, _ = original_img_np.shape
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
        
        # Overlay heatmap on original image
        # original_img_np must be float32 in range [0, 1]
        visualization = show_cam_on_image(original_img_np, grayscale_cam_resized, use_rgb=True)
        
        # Convert visualization (uint8, 0-255 RGB) to base64 JPEG
        pil_img = Image.fromarray(visualization)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        gradcam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return gradcam_base64
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        # Fallback to returning original image if Grad-CAM fails
        pil_img = Image.fromarray((original_img_np * 255).astype(np.uint8))
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
