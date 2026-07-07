import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
import torchvision.transforms as transforms
from model import SkinLesionModel, CLASSES
from train import ISICDataset

def analyze_bias_variance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Bias vs Variance Analysis on device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_file = os.path.join(data_dir, "HAM10000_metadata.csv")
    
    if not os.path.exists(csv_file):
        print("Error: Dataset not found locally. Cannot run offline analysis without dataset.")
        return
        
    full_dataset = ISICDataset(csv_file=csv_file, img_dir=data_dir)
    num_samples = len(full_dataset)
    
    train_indices, val_indices = train_test_split(
        list(range(num_samples)),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42
    )
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We evaluate both train and val subsets using the SAME validation transform
    # because we want to measure the model's true performance on the data distribution
    # without training-time augmentations (like random erasing/flipping) muddying the accuracy.
    
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.subset)
            
    full_dataset.transform = None
    
    raw_train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    raw_val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataset = TransformSubset(raw_train_subset, transform=val_transform)
    val_dataset = TransformSubset(raw_val_subset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = SkinLesionModel(num_classes=len(CLASSES))
    checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Error: model.pth not found. Cannot evaluate.")
        return
        
    model = model.to(device)
    model.eval()
    
    def evaluate_split(loader, split_name):
        correct = 0
        total = 0
        with torch.no_grad():
            tqdm_loader = tqdm(loader, desc=f"Evaluating {split_name}")
            for images, labels in tqdm_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                # Apply the same 0.2 threshold override for Melanoma as training
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)
                melanoma_mask = probs[:, 0] > 0.2
                predicted[melanoma_mask] = 0
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                tqdm_loader.set_postfix({'acc': correct/total})
        return correct / total
        
    print("\nEvaluating Training Set Accuracy (to measure Bias)...")
    train_acc = evaluate_split(train_loader, "Train Split")
    
    print("\nEvaluating Validation Set Accuracy (to measure Variance)...")
    val_acc = evaluate_split(val_loader, "Validation Split")
    
    acc_gap = train_acc - val_acc
    
    print("\n" + "="*40)
    print("BIAS VS VARIANCE DIAGNOSIS (OFFLINE)")
    print("="*40)
    print(f"Training Set Accuracy   : {train_acc*100:.2f}%")
    print(f"Validation Set Accuracy : {val_acc*100:.2f}%")
    print(f"Accuracy Gap (Train-Val): {acc_gap*100:.2f}%")
    print("-" * 40)
    
    if train_acc < 0.80:
        print("Diagnosis: HIGH BIAS (Underfitting)")
        print("Observation: The model fails to accurately capture the training data.")
        print("Recommendation: Use a more complex model (EfficientNetV2-M), train longer, or reduce regularization.")
    elif acc_gap > 0.10:
        print("Diagnosis: HIGH VARIANCE (Overfitting)")
        print("Observation: The model performs significantly better on training data than unseen validation data.")
        print("Recommendation: Add more data, apply stronger data augmentation, or increase dropout/weight decay.")
    else:
        print("Diagnosis: GOOD FIT (Balanced)")
        print("Observation: The model has low bias and generalizes well to unseen data (low variance).")
        print("Recommendation: Model is optimal for current data constraints.")

if __name__ == "__main__":
    analyze_bias_variance()
