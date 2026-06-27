import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(__file__))
from model import SkinLesionModel, CLASSES

# Mapping from HAM10000/ISIC csv diagnosis string to model class index
DIAGNOSIS_MAP = {
    'melanoma': 0,
    'mel': 0,
    'nevus': 1,
    'nv': 1,
    'melanocytic nevus': 1,
    'basal cell carcinoma': 2,
    'bcc': 2,
    'actinic keratosis': 3,
    'akiec': 3,
    'seborrheic keratosis': 4,
    'bkl': 4,
    'dermatofibroma': 5,
    'df': 5,
    'vascular lesion': 6,
    'vasc': 6
}

class FocalLoss(nn.Module):
    """
    Focal Loss down-weights well-classified examples and focuses on hard,
    misclassified samples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        # alpha is class weights tensor [num_classes]
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1.0 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ISICDataset(Dataset):
    """
    Custom PyTorch Dataset for loading ISIC dermoscopy images.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Support HAM10000 column names
        if 'dx' in self.df.columns:
            self.df = self.df.rename(columns={'dx': 'diagnosis', 'image_id': 'image_name'})
        
        # Clean diagnosis values
        self.df['diagnosis'] = self.df['diagnosis'].astype(str).str.lower().str.strip()
        
        # Filter to target diagnoses
        valid_diagnoses = list(DIAGNOSIS_MAP.keys())
        self.df = self.df[self.df['diagnosis'].isin(valid_diagnoses)].reset_index(drop=True)
        
        # Map diagnoses to class index labels
        self.labels = self.df['diagnosis'].map(DIAGNOSIS_MAP).values
        self.image_names = self.df['image_name'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        # Search for .jpg files
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        # Support HAM10000 nested directories
        if not os.path.exists(img_path):
            part1 = os.path.join(self.img_dir, "HAM10000_images_part_1", f"{img_name}.jpg")
            part2 = os.path.join(self.img_dir, "HAM10000_images_part_2", f"{img_name}.jpg")
            if os.path.exists(part1):
                img_path = part1
            elif os.path.exists(part2):
                img_path = part2
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback to a plain placeholder if image file is corrupted or missing
            image = Image.new('RGB', (224, 224), (200, 200, 200))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def setup_dry_run_data(data_dir):
    """
    Creates dummy CSV and image assets for verifying the training pipeline.
    """
    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, "train")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create 14 dummy images representing different diagnoses
    dummy_diagnoses = [
        'melanoma', 'nevus', 'basal cell carcinoma', 'actinic keratosis', 
        'seborrheic keratosis', 'dermatofibroma', 'vascular lesion',
        'melanoma', 'nevus', 'basal cell carcinoma', 'actinic keratosis', 
        'seborrheic keratosis', 'dermatofibroma', 'vascular lesion'
    ]
    
    img_names = []
    for i, diag in enumerate(dummy_diagnoses):
        name = f"ISIC_DUMMY_{i:03d}"
        img_names.append(name)
        
        # Create a simple color block image
        img = Image.new('RGB', (256, 256), color=(i * 15, 255 - i * 15, (i * 30) % 255))
        img.save(os.path.join(img_dir, f"{name}.jpg"))
        
    # Write a train_dummy.csv
    csv_df = pd.DataFrame({
        'image_name': img_names,
        'diagnosis': dummy_diagnoses,
        'target': [1 if diag in ['melanoma', 'basal cell carcinoma'] else 0 for diag in dummy_diagnoses]
    })
    csv_path = os.path.join(data_dir, "train_dummy.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Created dry-run dummy data at {csv_path} and {img_dir}/")
    return csv_path, img_dir

def train_model(args):
    # Set up devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    if args.dry_run:
        csv_file, img_dir = setup_dry_run_data(data_dir)
        epochs = 2
        batch_size = 2
        validation_split = 0.5
    else:
        csv_file = os.path.join(data_dir, "HAM10000_metadata.csv")
        img_dir = data_dir
        epochs = 20
        batch_size = 32
        validation_split = 0.2
        
        if not os.path.exists(csv_file):
            print(f"Error: Dataset files not found at {csv_file}.")
            print("Please run 'python setup_and_download.py <username>' first to retrieve the dataset.")
            sys.exit(1)
            
    # Load dataset
    full_dataset = ISICDataset(csv_file=csv_file, img_dir=img_dir)
    num_samples = len(full_dataset)
    print(f"Total mapped dataset samples: {num_samples}")
    
    if num_samples == 0:
        print("Error: No samples found matching target classes.")
        sys.exit(1)
        
    # Split into train and validation splits
    from sklearn.model_selection import train_test_split
    
    # Stratified split to ensure all classes are proportionally represented in train and val
    train_indices, val_indices = train_test_split(
        list(range(num_samples)),
        test_size=validation_split,
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # Setup data augmentation transforms (as outlined in README)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) # Cutout regularization
    ])

    # --- FIX: Stronger augmentation for melanoma class ---
    strong_melanoma_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90), # Stronger rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added affine
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0) # Stronger cutout
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Wrapper to correctly isolate transforms for train/val subsets
    class TransformSubset(Dataset):
        def __init__(self, subset, transform=None, strong_transform=None, target_label=None):
            self.subset = subset
            self.transform = transform
            self.strong_transform = strong_transform
            self.target_label = target_label

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.strong_transform and label == self.target_label:
                image = self.strong_transform(image)
            elif self.transform:
                image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.subset)

    # Prevent base dataset from applying its own transforms
    full_dataset.transform = None

    # Subclass datasets
    raw_train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    raw_val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply specific isolated transforms
    train_dataset = TransformSubset(
        raw_train_subset, 
        transform=train_transform, 
        strong_transform=strong_melanoma_transform, 
        target_label=0 # Melanoma index
    )
    val_dataset = TransformSubset(raw_val_subset, transform=val_transform)
    
    # Calculate sampler weights for the training split (WeightedRandomSampler)
    train_labels = full_dataset.labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=len(CLASSES))
    print(f"Train split class counts: {class_counts}")
    
    # Handle potentially zero classes in dry-run
    class_counts_safe = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts_safe
    # Set weights of empty classes back to 0
    class_weights[class_counts == 0] = 0.0
    
    # --- FIX: Boost minority classes by 5x (everything except the most frequent class) ---
    majority_class_idx = np.argmax(class_counts)
    for i in range(len(class_weights)):
        if i != majority_class_idx:
            class_weights[i] *= 5.0
    
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model Init
    model = SkinLesionModel(num_classes=len(CLASSES))
    model = model.to(device)
    
    # Loss configuration with class weight balance for Focal Loss
    # --- FIX: Boost melanoma class weight by 5x in focal loss ---
    alpha_loss = class_weights.copy()
    alpha_loss[0] *= 5.0
    
    # Normalize class weights
    alpha_loss = alpha_loss / np.sum(alpha_loss)
    
    # --- FIX: Increase focal loss gamma from 2.0 to 4.0 ---
    criterion = FocalLoss(alpha=alpha_loss, gamma=4.0).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_auc = 0.0
    checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pth")
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # --- FIX: Apply 0.2 classification threshold for Melanoma ---
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            melanoma_mask = probs[:, 0] > 0.2
            predicted[melanoma_mask] = 0
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_loader_tqdm.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_loader_tqdm:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                # --- FIX: Apply 0.2 classification threshold for Melanoma ---
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)
                melanoma_mask = probs[:, 0] > 0.2
                predicted[melanoma_mask] = 0
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Save predictions for multi-class AUC evaluation
                probs = torch.softmax(outputs, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
                
                val_loader_tqdm.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Calculate Multi-Class AUC-ROC
        all_val_preds = np.concatenate(val_preds, axis=0)
        all_val_labels = np.concatenate(val_labels_list, axis=0)
        
        # Find which classes are present in the validation split
        present_classes = np.unique(all_val_labels)
        
        if len(present_classes) > 1:
            try:
                # If doing multi-class ROC-AUC, we evaluate on present classes or standard OVR
                # One-vs-rest AUC
                val_auc = roc_auc_score(
                    y_true=all_val_labels,
                    y_score=all_val_preds,
                    multi_class='ovr',
                    labels=list(range(len(CLASSES)))
                )
            except Exception:
                # Fallback if some classes have 0 labels
                val_auc = 0.0
        else:
            val_auc = 0.0
            
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.1f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc*100:.1f}% AUC: {val_auc:.4f}")
              
        # Checkpointing based on AUC
        if val_auc >= best_val_auc or args.dry_run:
            best_val_auc = val_auc
            
            # --- FIX: Prevent dry-run from overwriting production model ---
            save_path = checkpoint_path
            if args.dry_run:
                save_path = checkpoint_path.replace(".pth", "_dryrun.pth")
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc
            }, save_path)
            print(f"--> Saved best model checkpoint to {save_path}")
            
            import shutil
            # Save to Drive as backup
            drive_path = '/content/drive/MyDrive/dermavision/model.pth'
            if os.path.exists('/content/drive/MyDrive/dermavision/'):
                shutil.copy(save_path, drive_path)
                print(f"Backup saved to Google Drive: {drive_path}")
            
    print("\nTraining execution completed successfully!")
    if args.dry_run:
        print("Dry-run test complete. Cleaned up dummy files.")
        # Cleanup dummy csv and dummy images
        try:
            os.remove(csv_file)
            for f in os.listdir(img_dir):
                os.remove(os.path.join(img_dir, f))
            os.rmdir(img_dir)
        except Exception as e:
            print(f"Warning during dry-run cleanup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermaVision Model Training Script")
    parser.add_argument('--dry-run', action='store_true', help="Run quick 2-epoch test on local synthesized mock data")
    args = parser.parse_args()
    train_model(args)
