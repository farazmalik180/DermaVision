import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from model import load_model

def generate_forward_prop_animation(image_path, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    model = load_model(model_path).to(device)
    model.eval()
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    features = []
    names = []
    
    def get_features(name):
        def hook(model, input, output):
            # Calculate mean activation across channels
            mean_act = torch.mean(output.detach(), dim=1).squeeze(0).cpu().numpy()
            features.append(mean_act)
            names.append(name)
        return hook
        
    hooks = []
    # Register hook for every major block in features
    for i, block in enumerate(model.model.features):
        hooks.append(block.register_forward_hook(get_features(f'Block {i}')))
        
    print(f"Extracting features for animation...")
    with torch.no_grad():
        _ = model(input_tensor)
        
    for h in hooks:
        h.remove()
        
    # Generate Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # Initialize with first frame
    im = ax.imshow(features[0], cmap='viridis', animated=True)
    title = ax.set_title(names[0], fontsize=16)
    
    def update(frame):
        im.set_array(features[frame])
        title.set_text(f"Forward Propagation: {names[frame]}")
        return im, title
        
    ani = animation.FuncAnimation(fig, update, frames=len(features), interval=800, blit=True)
    
    output_path = os.path.join(os.path.dirname(__file__), "forward_prop.gif")
    ani.save(output_path, writer='pillow', fps=1.5)
    print(f"Saved animation to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(__file__), "clear_lesion.jpg"))
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), "model.pth"))
    args = parser.parse_args()
    
    generate_forward_prop_animation(args.image, args.model)
