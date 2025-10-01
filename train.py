"""
ToothFairy3 Model Training Script
Train a 3D U-Net for IAC (Inferior Alveolar Canal) segmentation with interactive click guidance.
"""

import os
import random
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.losses import DiceLoss, FocalLoss
from monai.networks.nets import UNet
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, MapTransform,
                              RandAdjustContrastd, RandGaussianNoised, Resized,
                              ScaleIntensityd, Spacingd, ToTensord)
from monai.utils import set_determinism
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# CONFIGURATION
#change the configs according to your hardware capabilities
    # Paths
    DATA_DIR = "/input" #change to your path
    CHECKPOINT_DIR = "model_checkpoints"
    VIZ_DIR = "visualizations"
    
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    NUM_CLASSES = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Data loading
    CACHE_RATE = 1.0
    NUM_WORKERS = 4
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Early stopping
    PATIENCE = 15
    
    # Image processing
    SPATIAL_SIZE = (384, 384, 192)
    PIXDIM = (0.3, 0.3, 0.3)
    
    # Click map parameters
    CLICK_SIGMA = 3
    MIN_CLICKS = 3
    MAX_CLICKS = 7

config = Config()

# CUSTOM TRANSFORMS

class AddIACClickMapsd(MapTransform):
    """
    Add interactive click maps for left and right IAC segmentation.
    Simulates user clicks along the canal structure.
    """
    
    def __init__(self, keys, sigma=3, min_clicks=3, max_clicks=7):
        super().__init__(keys)
        self.sigma = sigma
        self.min_clicks = min_clicks
        self.max_clicks = max_clicks

    def _sample_clicks_from_mask(self, mask):
        """Sample click points along the canal structure"""
        num_clicks = 5
        indices = np.argwhere(mask)
        
        if indices.size == 0:
            return []

        # Get coordinate ranges
        x_coords, y_coords, z_coords = indices[:, 0], indices[:, 1], indices[:, 2]
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Sample start and end points
        x_start = np.random.randint(x_min, min(x_min + 6, x_max + 1))
        x_end = np.random.randint(max(x_max - 5, x_min), x_max + 1)

        # Get valid x coordinates
        valid_x = [el for el in np.arange(x_min, x_max + 1) if np.any(mask[el])]
        sampled_x = np.linspace(x_start, x_end, num=num_clicks, dtype=int)
        sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

        # Add random jitter
        sampled_x += np.random.randint(-3, 4, size=num_clicks)
        sampled_x = np.clip(sampled_x, x_start, x_end)
        sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

        # Fix endpoints
        sampled_x[0], sampled_x[-1] = x_start, x_end
        if x_start not in valid_x:
            sampled_x[0] = x_min
        if x_end not in valid_x:
            sampled_x[-1] = x_max

        # Generate click points
        click_points = []
        for x in sampled_x:
            valid_points = indices[x_coords == x]
            y_center = np.median(valid_points[:, 1]).astype(int)
            z_center = np.median(valid_points[:, 2]).astype(int)
            
            # Try to place click within mask
            for _ in range(10):
                y = np.clip(y_center + np.random.randint(-3, 4), y_coords.min(), y_coords.max())
                z = np.clip(z_center + np.random.randint(-3, 4), z_coords.min(), z_coords.max())
                if mask[x, y, z]:
                    click_points.append((x, y, z))
                    break
            else:
                y, z = valid_points[random.randint(0, valid_points.shape[0] - 1), 1:]
                click_points.append((x, int(y), int(z)))

        return click_points

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        lbl = d["label"]

        # Separate left and right IAC masks
        left_mask = lbl[0] == 1
        right_mask = lbl[0] == 2

        # Sample clicks for each canal
        left_clicks = self._sample_clicks_from_mask(left_mask)
        right_clicks = self._sample_clicks_from_mask(right_mask)

        # Create click maps
        left_map = np.zeros_like(img[0], dtype=np.float32)
        right_map = np.zeros_like(img[0], dtype=np.float32)

        for x, y, z in left_clicks:
            left_map[x, y, z] = 1
        for x, y, z in right_clicks:
            right_map[x, y, z] = 1

        # Apply Gaussian smoothing
        left_map = gaussian_filter(left_map, sigma=self.sigma)
        right_map = gaussian_filter(right_map, sigma=self.sigma)

        # Stack image with click maps
        d["image"] = np.stack([img[0], left_map, right_map], axis=0)
        return d


# DATA PREPARATION


def get_transforms(is_train=True):
    """Get data transforms for training or validation"""
    
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config.PIXDIM, mode=("trilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=config.SPATIAL_SIZE, mode=("trilinear", "nearest")),
        AddIACClickMapsd(keys=["image", "label"], sigma=config.CLICK_SIGMA, 
                         min_clicks=config.MIN_CLICKS, max_clicks=config.MAX_CLICKS),
    ]
    
    # Add augmentation for training
    if is_train:
        base_transforms.extend([
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9, 1.1)),
        ])
    
    # Final conversions
    base_transforms.extend([
        CastToTyped(keys="label", dtype=np.int64),
        ToTensord(keys=["image", "label"]),
    ])
    
    return Compose(base_transforms)

def prepare_data():
    """Load and split data into train/validation sets"""
    
    images = sorted(glob(os.path.join(config.DATA_DIR, "imagesTr", "*.nii*")))
    labels = sorted(glob(os.path.join(config.DATA_DIR, "labelsTr", "*.nii*")))
    
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    train_data, val_data = train_test_split(
        data_dicts, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED, 
        shuffle=True
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_dataloaders(train_data, val_data):
    """Create MONAI data loaders with caching"""
    
    train_transforms = get_transforms(is_train=True)
    val_transforms = get_transforms(is_train=False)
    
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=config.CACHE_RATE)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=config.CACHE_RATE)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=pad_list_data_collate, 
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=pad_list_data_collate, 
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader


# MODEL AND LOSS

def create_model(device):
    """Create 3D U-Net model"""
    
    model = UNet(
        spatial_dims=3,
        in_channels=3,  # Image + 2 click maps
        out_channels=config.NUM_CLASSES,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="batch",
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def get_loss_functions():
    """Create combined loss function"""
    
    dice_loss = DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
    )
    
    focal_loss = FocalLoss(
        include_background=True,
        to_onehot_y=True,
        use_softmax=True,
        alpha=0.25,
        gamma=2.0,
    )
    
    def combined_loss(pred, target):
        return dice_loss(pred, target) + 0.5 * focal_loss(pred, target)
    
    return combined_loss

# METRICS

def calculate_dice_per_class(y_true, y_pred, num_classes=3):
    """Calculate Dice coefficient for each class (excluding background)"""
    
    dice_scores = []
    
    for class_id in range(1, num_classes):
        y_true_class = (y_true == class_id).astype(np.float32)
        y_pred_class = (y_pred == class_id).astype(np.float32)
        
        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class) + np.sum(y_pred_class)
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union
        
        dice_scores.append(dice)
    
    return np.array(dice_scores)

# TRAINING


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    
    model.train()
    epoch_loss = 0
    step = 0
    
    with tqdm(loader, desc="Training") as pbar:
        for batch_data in pbar:
            inputs = batch_data["image"].to(device)
            targets = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return epoch_loss / step

def validate_epoch(model, loader, loss_fn, device):
    """Validate for one epoch"""
    
    model.eval()
    epoch_loss = 0
    step = 0
    dice_scores_all = []
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for batch_data in pbar:
                inputs = batch_data["image"].to(device)
                targets = batch_data["label"].to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                epoch_loss += loss.item()
                step += 1
                
                # Calculate Dice scores
                outputs_softmax = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs_softmax, dim=1)
                
                preds_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                for i in range(preds_np.shape[0]):
                    dice_per_class = calculate_dice_per_class(targets_np[i, 0], preds_np[i])
                    dice_scores_all.append(dice_per_class)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    dice_scores_all = np.stack(dice_scores_all, axis=0)
    mean_dice = dice_scores_all.mean()
    
    return epoch_loss / step, mean_dice

# VISUALIZATION

def visualize_predictions(model, data_loader, device, epoch, sample_idx=0):
    """Visualize predictions on sagittal, coronal, and axial slices"""
    
    os.makedirs(config.VIZ_DIR, exist_ok=True)
    
    colors = ['black', 'red', 'blue']
    iac_cmap = ListedColormap(colors)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if i == sample_idx:
                inputs = batch_data["image"].to(device)
                targets = batch_data["label"].to(device)
                
                outputs = model(inputs)
                outputs_softmax = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs_softmax, dim=1)
                
                inputs_cpu = inputs[0, 0].cpu().numpy()
                targets_cpu = targets[0, 0].cpu().numpy()
                predictions_cpu = predictions[0].cpu().numpy()
                
                H, W, D = inputs_cpu.shape
                
                slices = {
                    'Sagittal': (inputs_cpu[H//2, :, :], targets_cpu[H//2, :, :], predictions_cpu[H//2, :, :]),
                    'Coronal': (inputs_cpu[:, W//2, :], targets_cpu[:, W//2, :], predictions_cpu[:, W//2, :]),
                    'Axial': (inputs_cpu[:, :, D//2], targets_cpu[:, :, D//2], predictions_cpu[:, :, D//2])
                }
                
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                fig.suptitle(f'Epoch {epoch+1} - Sample {sample_idx}', fontsize=16, fontweight='bold')
                
                for row, (view_name, (img_slice, gt_slice, pred_slice)) in enumerate(slices.items()):
                    axes[row, 0].imshow(img_slice, cmap='gray')
                    axes[row, 0].set_title(f'{view_name} - Input')
                    axes[row, 0].axis('off')
                    
                    axes[row, 1].imshow(gt_slice, cmap=iac_cmap, vmin=0, vmax=2, alpha=0.8)
                    axes[row, 1].imshow(img_slice, cmap='gray', alpha=0.3)
                    axes[row, 1].set_title(f'{view_name} - Ground Truth')
                    axes[row, 1].axis('off')
                    
                    axes[row, 2].imshow(pred_slice, cmap=iac_cmap, vmin=0, vmax=2, alpha=0.8)
                    axes[row, 2].imshow(img_slice, cmap='gray', alpha=0.3)
                    axes[row, 2].set_title(f'{view_name} - Prediction')
                    axes[row, 2].axis('off')
                
                dice_scores = calculate_dice_per_class(targets_cpu, predictions_cpu)
                dice_text = f"Dice - Class 1: {dice_scores[0]:.3f}, Class 2: {dice_scores[1]:.3f}, Mean: {dice_scores.mean():.3f}"
                fig.text(0.5, 0.02, dice_text, ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, bottom=0.08)
                
                save_path = os.path.join(config.VIZ_DIR, f'epoch_{epoch+1:03d}_sample_{sample_idx}.png')
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                break

def plot_training_history(train_losses, val_losses, val_dice_scores):
    """Plot training curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_dice_scores, label='Val Dice Score', color='green')
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

# MAIN TRAINING LOOP

def train_model(model, train_loader, val_loader, loss_fn, device):
    """Main training loop with early stopping"""
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_dice = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    print("Starting training...")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate_epoch(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loader, device, epoch, sample_idx=0)
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
            
            print(f"✓ New best model saved! Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best validation Loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, val_dice_scores

# MAIN

def main():
    """Main training pipeline"""
    
    # Setup
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=config.RANDOM_SEED)
    print(f"Using device: {device}")
    
    # Prepare data
    train_data, val_data = prepare_data()
    train_loader, val_loader = create_dataloaders(train_data, val_data)
    
    # Create model and loss
    model = create_model(device)
    loss_fn = get_loss_functions()
    
    # Train
    train_losses, val_losses, val_dice_scores = train_model(
        model, train_loader, val_loader, loss_fn, device
    )
    
    # Plot results
    plot_training_history(train_losses, val_losses, val_dice_scores)
    
    print("\n✓ Training completed successfully!")
    print("Saved files:")
    print("  - best_model.pth: Best model checkpoint")
    print("  - training_history.png: Training curves")
    print(f"  - {config.VIZ_DIR}/: Visualization outputs")

if __name__ == "__main__":
    main()
