# ===================================================================
# PRETRAINED U-NET DENGAN CONFUSION MATRIX
# ===================================================================

# Install library
!pip install segmentation-models-pytorch albumentations seaborn

import os
import numpy as np
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import wandb

# ===================================================================
# 1. CONFIGURATION
# ===================================================================
CONFIG = {
    # Data
    "train_image_dir": "./dataset_masks/train/images",
    "train_mask_dir": "./dataset_masks/train/label_masks",
    "val_image_dir": "./dataset_masks/valid/images",
    "val_mask_dir": "./dataset_masks/valid/label_masks",
    
    # Training
    "image_size": 512,
    "batch_size": 8,
    "val_batch_size": 4,
    "epochs": 100,
    "num_classes": 5,
    "class_names": ["Background", "Class1", "Class2", "Class3", "Class4"],
    
    # Model - PRETRAINED U-NET
    "encoder_name": "efficientnet-b0",
    "encoder_weights": "imagenet",
    
    # Optimization
    "lr_encoder": 1e-5,
    "lr_decoder": 1e-4,
    "weight_decay": 1e-4,
    
    # Misc
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "num_workers": 4,
    "save_path": "best_pretrained_unetv4.pth",
    "confusion_matrix_dir": "./confusion_matrices"
}

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# Create confusion matrix directory
os.makedirs(CONFIG["confusion_matrix_dir"], exist_ok=True)

# ===================================================================
# 2. DATASET
# ===================================================================
class FungiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, num_classes, is_train=True):
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_train = is_train
        
        # Find matching image-mask pairs
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        valid_pairs = []
        
        for mask_path in mask_paths:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            if os.path.exists(image_path):
                valid_pairs.append((image_path, mask_path))
        
        if len(valid_pairs) == 0:
            raise ValueError(f"No valid pairs found in {image_dir} and {mask_dir}")
        
        self.image_paths, self.mask_paths = zip(*valid_pairs)
        print(f"✅ {'Train' if is_train else 'Val'}: {len(self.image_paths)} images")
        
        # Define transforms
        if is_train:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = torch.from_numpy(transformed["mask"]).long()
        
        return image, mask

# ===================================================================
# 3. MODEL
# ===================================================================
def create_pretrained_unet(config):
    model = smp.Unet(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=3,
        classes=config["num_classes"],
        activation=None,
    )
    return model

# ===================================================================
# 4. LOSS FUNCTION
# ===================================================================
class CombinedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return 0.5 * ce + 0.5 * dice

# ===================================================================
# 5. CONFUSION MATRIX PLOTTING
# ===================================================================
def plot_confusion_matrix(cm, class_names, epoch, save_dir, normalize=False):
    """
    Plot and save confusion matrix
    
    Args:
        cm: confusion matrix array
        class_names: list of class names
        epoch: current epoch number
        save_dir: directory to save the plot
        normalize: whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'Normalized Confusion Matrix - Epoch {epoch}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - Epoch {epoch}'
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add statistics text
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    stats_text = f'Total Samples: {int(total_samples):,}\n'
    stats_text += f'Correct: {int(correct_predictions):,}\n'
    stats_text += f'Accuracy: {accuracy:.2%}'
    
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save normalized version
    if not normalize:
        save_path_norm = plot_confusion_matrix(cm, class_names, epoch, save_dir, normalize=True)
        return save_path, save_path_norm
    
    return save_path

def plot_per_class_metrics(cm, class_names, epoch, save_dir):
    """
    Plot per-class precision, recall, and F1-score from confusion matrix
    """
    num_classes = len(class_names)
    
    # Calculate per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(num_classes):
        # Precision = TP / (TP + FP)
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision)
        
        # Recall = TP / (TP + FN)
        fn = cm[i, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_per_class.append(f1)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(num_classes)
    width = 0.6
    
    # Precision
    bars1 = axes[0].bar(x, precision_per_class, width, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Precision', fontweight='bold')
    axes[0].set_title('Per-Class Precision', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{precision_per_class[i]:.2%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Recall
    bars2 = axes[1].bar(x, recall_per_class, width, color='coral', alpha=0.8)
    axes[1].set_ylabel('Recall', fontweight='bold')
    axes[1].set_title('Per-Class Recall', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{recall_per_class[i]:.2%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # F1-Score
    bars3 = axes[2].bar(x, f1_per_class, width, color='mediumseagreen', alpha=0.8)
    axes[2].set_ylabel('F1-Score', fontweight='bold')
    axes[2].set_title('Per-Class F1-Score', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1_per_class[i]:.2%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Per-Class Metrics - Epoch {epoch}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'per_class_metrics_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path, precision_per_class, recall_per_class, f1_per_class

# ===================================================================
# 6. METRICS CALCULATION
# ===================================================================
def calculate_metrics(pred, target, num_classes):
    """Calculate comprehensive metrics including confusion matrix"""
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Overall metrics
    precision = precision_score(target_flat, pred_flat, average='macro', zero_division=0)
    recall = recall_score(target_flat, pred_flat, average='macro', zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='macro', zero_division=0)
    accuracy = accuracy_score(target_flat, pred_flat)
    
    # Confusion Matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    
    # Per-class IoU
    iou_scores = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        
        if union == 0:
            iou_scores.append(float('nan'))
        else:
            iou_scores.append(intersection / union)
    
    miou = np.nanmean(iou_scores)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'miou': miou,
        'per_class_iou': iou_scores,
        'confusion_matrix': cm
    }

# ===================================================================
# 7. TRAINING LOOP
# ===================================================================
def train_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_targets.append(masks)
        
        pbar.set_postfix({'loss': loss.item()})
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

# ===================================================================
# 8. VALIDATION LOOP
# ===================================================================
def validate_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(masks)
            
            pbar.set_postfix({'loss': loss.item()})
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

# ===================================================================
# 9. VISUALIZATION
# ===================================================================
def visualize_predictions(model, dataset, device, num_samples=3, save_path='predictions.png'):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.cpu().numpy()
        
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='tab10', vmin=0, vmax=CONFIG["num_classes"]-1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='tab10', vmin=0, vmax=CONFIG["num_classes"]-1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

# ===================================================================
# 10. MAIN TRAINING
# ===================================================================
def main():
    wandb.init(
        project="segmentasi-jamurunet",
        name=f"pretrained-unet-{CONFIG['encoder_name']}-with-cm",
        config=CONFIG
    )
    
    print(f"\n{'='*70}")
    print(f"🚀 PRETRAINED U-NET TRAINING WITH CONFUSION MATRIX")
    print(f"{'='*70}")
    print(f"📦 Encoder: {CONFIG['encoder_name']}")
    print(f"🎯 Pretrained: {CONFIG['encoder_weights']}")
    print(f"💻 Device: {CONFIG['device']}")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_dataset = FungiDataset(
        CONFIG["train_image_dir"], CONFIG["train_mask_dir"],
        CONFIG["image_size"], CONFIG["num_classes"], is_train=True
    )
    val_dataset = FungiDataset(
        CONFIG["val_image_dir"], CONFIG["val_mask_dir"],
        CONFIG["image_size"], CONFIG["num_classes"], is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["val_batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    
    # Create model
    model = create_pretrained_unet(CONFIG)
    model = model.to(CONFIG["device"])
    
    print(f"✅ Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters\n")
    
    # Optimizer
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': CONFIG["lr_encoder"]},
        {'params': decoder_params, 'lr': CONFIG["lr_decoder"]}
    ], weight_decay=CONFIG["weight_decay"])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = CombinedLoss()
    
    # Training loop
    best_val_miou = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            CONFIG["device"], CONFIG["num_classes"]
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion,
            CONFIG["device"], CONFIG["num_classes"]
        )
        
        scheduler.step()
        
        # Print metrics
        print(f"\n📊 Training:")
        print(f"   Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | mIoU: {train_metrics['miou']:.4f}")
        
        print(f"\n📊 Validation:")
        print(f"   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | mIoU: {val_metrics['miou']:.4f}")
        
        print(f"\n📈 Per-Class IoU:")
        for cls_idx, cls_name in enumerate(CONFIG["class_names"]):
            iou = val_metrics['per_class_iou'][cls_idx]
            if not np.isnan(iou):
                print(f"   {cls_name}: {iou:.4f}")
        
        # Plot confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Generating confusion matrix...")
            
            # Plot confusion matrices
            cm_path, cm_norm_path = plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                CONFIG["class_names"],
                epoch + 1,
                CONFIG["confusion_matrix_dir"]
            )
            
            # Plot per-class metrics
            metrics_path, prec_list, rec_list, f1_list = plot_per_class_metrics(
                val_metrics['confusion_matrix'],
                CONFIG["class_names"],
                epoch + 1,
                CONFIG["confusion_matrix_dir"]
            )
            
            # Log to wandb
            wandb.log({
                "confusion_matrix": wandb.Image(cm_path),
                "confusion_matrix_normalized": wandb.Image(cm_norm_path),
                "per_class_metrics": wandb.Image(metrics_path)
            })
            
            print(f"   ✅ Confusion matrix saved to: {cm_path}")
        
        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['accuracy'],
            "train/miou": train_metrics['miou'],
            "train/f1": train_metrics['f1'],
            "train/precision": train_metrics['precision'],
            "train/recall": train_metrics['recall'],
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/miou": val_metrics['miou'],
            "val/f1": val_metrics['f1'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Add per-class IoU
        for cls_idx, cls_name in enumerate(CONFIG["class_names"]):
            iou = val_metrics['per_class_iou'][cls_idx]
            if not np.isnan(iou):
                log_dict[f"val/IoU_{cls_name}"] = iou
        
        wandb.log(log_dict)
        
        # Save best model
        if val_metrics['miou'] > best_val_miou:
            best_val_miou = val_metrics['miou']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_metrics['miou'],
                'confusion_matrix': val_metrics['confusion_matrix'],
                'config': CONFIG
            }, CONFIG["save_path"])
            
            print(f"\n✅ New best model! mIoU: {best_val_miou:.4f}")
            
            # Visualize predictions
            viz_path = visualize_predictions(model, val_dataset, CONFIG["device"])
            wandb.log({"predictions": wandb.Image(viz_path)})
        else:
            patience_counter += 1
            print(f"\n⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Final confusion matrix
    print(f"\n📊 Generating final confusion matrix...")
    cm_path, cm_norm_path = plot_confusion_matrix(
        val_metrics['confusion_matrix'],
        CONFIG["class_names"],
        "FINAL",
        CONFIG["confusion_matrix_dir"]
    )
    
    metrics_path, _, _, _ = plot_per_class_metrics(
        val_metrics['confusion_matrix'],
        CONFIG["class_names"],
        "FINAL",
        CONFIG["confusion_matrix_dir"]
    )
    
    wandb.log({
        "final/confusion_matrix": wandb.Image(cm_path),
        "final/confusion_matrix_normalized": wandb.Image(cm_norm_path),
        "final/per_class_metrics": wandb.Image(metrics_path)
    })
    
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"🏆 Best Validation mIoU: {best_val_miou:.4f}")
    print(f"💾 Model: {CONFIG['save_path']}")
    print(f"📊 Confusion matrices: {CONFIG['confusion_matrix_dir']}")
    print(f"{'='*70}\n")
    
    wandb.finish()

# ===================================================================
# 11. RUN
# ===================================================================
if __name__ == "__main__":
    main()
