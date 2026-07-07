import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(__file__))
from model import SkinLesionModel, load_model

def visualize_layer_features(image_path, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # 1. Load Model
    model = load_model(model_path)
    model = model.to(device)
    model.eval()
    
    # 2. Load and Preprocess Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    img = Image.open(image_path).convert('RGB')
    
    # Same transforms as validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 3. Setup Hooks to extract feature maps
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
        
    # EfficientNetV2-S backbone layers
    # features[0]: Stem
    # features[2]: Fused-MBConv (Shallow)
    # features[5]: MBConv (Middle)
    # features[7]: Final Conv (Deep)
    
    hooks = []
    try:
        hooks.append(model.model.features[0].register_forward_hook(get_features('Stem (Layer 0)')))
        hooks.append(model.model.features[2].register_forward_hook(get_features('Fused-MBConv (Layer 2)')))
        hooks.append(model.model.features[5].register_forward_hook(get_features('MBConv (Layer 5)')))
        hooks.append(model.model.features[7].register_forward_hook(get_features('Final Conv (Layer 7)')))
    except Exception as e:
        print(f"Error registering hooks: {e}")
        return
        
    # 4. Forward Pass
    print(f"Extracting features for {os.path.basename(image_path)}...")
    with torch.no_grad():
        _ = model(input_tensor)
        
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # 5. Plotting
    num_layers = len(features)
    fig, axes = plt.subplots(num_layers, 5, figsize=(15, 3 * num_layers))
    fig.suptitle('Layer-wise Feature Map Visualization', fontsize=16)
    
    layer_names = list(features.keys())
    
    for i, name in enumerate(layer_names):
        fmap = features[name].cpu().squeeze(0) # [Channels, Height, Width]
        
        # We will plot the mean activation across all channels as the first image
        # and then 4 random channels for variety.
        
        mean_act = torch.mean(fmap, dim=0).numpy()
        axes[i, 0].imshow(mean_act, cmap='viridis')
        axes[i, 0].set_title(f'{name}\nMean Activation')
        axes[i, 0].axis('off')
        
        # Pick 4 random channels (seeded for consistency)
        np.random.seed(42)
        num_channels = fmap.shape[0]
        random_channels = np.random.choice(num_channels, 4, replace=False)
        
        for j, ch in enumerate(random_channels):
            axes[i, j+1].imshow(fmap[ch].numpy(), cmap='viridis')
            axes[i, j+1].set_title(f'Channel {ch}')
            axes[i, j+1].axis('off')
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = os.path.join(os.path.dirname(__file__), "layer_features.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved feature visualization to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Layer-wise CNN Features")
    parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(__file__), "clear_lesion.jpg"), help="Path to input image")
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), "model.pth"), help="Path to model weights")
    args = parser.parse_args()
    
    visualize_layer_features(args.image, args.model)
