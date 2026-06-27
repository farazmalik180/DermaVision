import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

sys.path.append(os.path.dirname(__file__))
import torchvision.transforms as transforms
from model import SkinLesionModel, CLASSES
from train import ISICDataset

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_file = os.path.join(data_dir, "HAM10000_metadata.csv")
    
    if not os.path.exists(csv_file):
        print("Dataset not found locally.")
        return
        
    full_dataset = ISICDataset(csv_file=csv_file, img_dir=data_dir)
    num_samples = len(full_dataset)
    
    train_indices, val_indices = train_test_split(
        list(range(num_samples)),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # Wrapper to correctly isolate transforms
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
    raw_val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset = TransformSubset(raw_val_subset, transform=val_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = SkinLesionModel(num_classes=len(CLASSES))
    checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if os.path.exists(checkpoint_path):
        # Allow weights_only=False to load legacy saves if needed, but dicts should load fine
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("model.pth not found!")
        return
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    
    sensitivities = []
    specificities = []
    
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sens)
        specificities.append(spec)
        
        print(f"Class: {CLASSES[i]}")
        print(f"  Sensitivity: {sens*100:.2f}%")
        print(f"  Specificity: {spec*100:.2f}%")
        
    print("\nOverall Macro Average:")
    print(f"Macro Sensitivity: {np.mean(sensitivities)*100:.2f}%")
    print(f"Macro Specificity: {np.mean(specificities)*100:.2f}%")
    
    # Melanoma specific (Class 0)
    print("\nMelanoma (Malignant) Specific:")
    print(f"Sensitivity: {sensitivities[0]*100:.2f}%")
    print(f"Specificity: {specificities[0]*100:.2f}%")

    target_names = [c["name"] for c in CLASSES]
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("\n--- Confusion Matrix ---")
    print(cm)
    
    print("\n--- Per-Class AUC-ROC ---")
    all_probs = np.array(all_probs)
    for i, name in enumerate(target_names):
        try:
            auc = roc_auc_score(np.array(all_labels) == i, all_probs[:, i])
            print(f"{name}: {auc:.4f}")
        except ValueError:
            print(f"{name}: N/A (Only one class in true labels)")

if __name__ == "__main__":
    evaluate()
