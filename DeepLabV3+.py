import os
import numpy as np
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
import gc
import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# =============================================
# LOSS FUNCTIONS (Embedded)
# =============================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance.
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
                alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss untuk segmentation.
    Formula: DL = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth=1.0, ignore_index=None, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        B, C, H, W = inputs.shape
        inputs = F.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        inputs = inputs.view(B, C, -1)
        targets_one_hot = targets_one_hot.view(B, C, -1)

        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice_coef = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coef

        if self.ignore_index is not None:
            mask = torch.ones(C, device=inputs.device)
            mask[self.ignore_index] = 0
            dice_loss = dice_loss * mask.unsqueeze(0)
            dice_loss = dice_loss.sum(dim=1) / mask.sum()
        else:
            dice_loss = dice_loss.mean(dim=1)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss yang menggabungkan multiple loss functions.
    """
    def __init__(self,
                 loss_weights={'ce': 1.0},
                 focal_params=None,
                 dice_params=None,
                 label_smoothing=0.0):
        super(CombinedLoss, self).__init__()

        self.loss_weights = loss_weights
        self.losses = {}

        if 'ce' in loss_weights and loss_weights['ce'] > 0:
            self.losses['ce'] = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if 'focal' in loss_weights and loss_weights['focal'] > 0:
            if focal_params is None:
                focal_params = {'alpha': None, 'gamma': 2.0}
            self.losses['focal'] = FocalLoss(
                alpha=focal_params.get('alpha', None),
                gamma=focal_params.get('gamma', 2.0),
                label_smoothing=label_smoothing
            )

        if 'dice' in loss_weights and loss_weights['dice'] > 0:
            if dice_params is None:
                dice_params = {'smooth': 1.0, 'ignore_index': None}
            self.losses['dice'] = DiceLoss(
                smooth=dice_params.get('smooth', 1.0),
                ignore_index=dice_params.get('ignore_index', None)
            )

        print("📊 Combined Loss Configuration:")
        for loss_name, weight in loss_weights.items():
            if weight > 0:
                print(f"   - {loss_name.upper()}: weight={weight}")

    def forward(self, inputs, targets):
        total_loss = 0.0
        loss_components = {}

        for loss_name, loss_fn in self.losses.items():
            weight = self.loss_weights[loss_name]
            if weight > 0:
                loss_value = loss_fn(inputs, targets)
                total_loss += weight * loss_value
                loss_components[loss_name] = loss_value.item()

        return total_loss, loss_components


def get_loss_function(loss_config):
    """Factory function to create loss function based on config."""
    loss_type = loss_config.get('type', 'ce').lower()

    if loss_type == 'ce':
        print("🎯 Using Cross Entropy Loss")
        return nn.CrossEntropyLoss(
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    elif loss_type == 'focal':
        print("🎯 Using Focal Loss")
        return FocalLoss(
            alpha=loss_config.get('alpha', None),
            gamma=loss_config.get('gamma', 2.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    elif loss_type == 'dice':
        print("🎯 Using Dice Loss")
        return DiceLoss(
            smooth=loss_config.get('smooth', 1.0),
            ignore_index=loss_config.get('ignore_index', None)
        )

    elif loss_type == 'combined':
        print("🎯 Using Combined Loss")
        return CombinedLoss(
            loss_weights=loss_config.get('weights', {'ce': 1.0}),
            focal_params=loss_config.get('focal_params', None),
            dice_params=loss_config.get('dice_params', None),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================
# CONFIG & REPRODUCIBILITY
# =============================================

CONFIG = {
    "image_size": 512,
    "batch_size": 8,
    "val_batch_size": 1,
    "epochs": 100,
    "lr": 5e-5,
    "num_classes": 5,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "gradient_accumulation_steps": 2,
    "weight_decay": 5e-4,
    "dropout_rate": 0.3,
    "use_mixup": True,
    "mixup_alpha": 0.2,
    "early_stopping_patience": 15,
    "min_delta": 0.001,

    # ⭐ LOSS CONFIGURATION - Focal + Dice (RECOMMENDED)
    "loss_config": {
        "type": "combined",
        "weights": {"focal": 0.4, "dice": 0.6},
        "focal_params": {
            "alpha": [0.5, 2.0, 2.5, 1.8, 3.0],  # [Background, Jamur, Foxing, Noda, Korosi]
            "gamma": 2.0
        },
        "dice_params": {
            "smooth": 1.0,
            "ignore_index": 0  # Ignore background
        },
        "label_smoothing": 0.2
    },
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
    print(f"CUDA Memory: {torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")

wandb.init(
    project="segmentasi-jamur-manuscript",
    name="deeplabv3plus-focal-dice-improved",
    config=CONFIG
)

# =============================================
# EARLY STOPPING
# =============================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_miou, epoch):
        score = val_miou

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f'✅ Initial best validation mIoU: {self.best_score:.4f}')
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'⚠️  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'🛑 Early stopping triggered! Best mIoU: {self.best_score:.4f} at epoch {self.best_epoch}')
        else:
            improvement = score - self.best_score
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f'✅ Validation mIoU improved by {improvement:.4f}! New best: {self.best_score:.4f}')
            self.counter = 0

        return self.early_stop

# =============================================
# DATASET
# =============================================

class FungiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.image_paths = []
        self.is_train = is_train

        valid_pairs = []
        for mask_path in self.mask_paths:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            if os.path.exists(image_path):
                valid_pairs.append((image_path, mask_path))

        if not valid_pairs:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

        self.image_paths, self.mask_paths = zip(*valid_pairs)
        print(f"✅ Loaded {len(self.image_paths)} image-mask pairs for {'training' if is_train else 'validation'}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            mask = Image.open(self.mask_paths[idx]).convert("L")

            if self.is_train:
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)

                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)

                if random.random() > 0.5:
                    angle = random.uniform(-30, 30)
                    image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
                    mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

                if random.random() > 0.5:
                    scale = random.uniform(0.8, 1.2)
                    translate = (random.randint(-50, 50), random.randint(-50, 50))
                    image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0,
                                    interpolation=T.InterpolationMode.BILINEAR)
                    mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0,
                                   interpolation=T.InterpolationMode.NEAREST)

                if random.random() > 0.5:
                    image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
                if random.random() > 0.5:
                    image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
                if random.random() > 0.5:
                    image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
                if random.random() > 0.5:
                    image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

                if random.random() > 0.7:
                    image = TF.gaussian_blur(image, kernel_size=5)

                if random.random() > 0.8:
                    image = TF.rgb_to_grayscale(image, num_output_channels=3)

            image = TF.resize(image, [CONFIG["image_size"], CONFIG["image_size"]],
                            interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [CONFIG["image_size"], CONFIG["image_size"]],
                           interpolation=T.InterpolationMode.NEAREST)

            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            mask = torch.from_numpy(np.array(mask)).long()
            mask = torch.clamp(mask, 0, CONFIG["num_classes"] - 1)

            return image, mask

        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            dummy_image = torch.zeros(3, CONFIG["image_size"], CONFIG["image_size"])
            dummy_mask = torch.zeros(CONFIG["image_size"], CONFIG["image_size"]).long()
            return dummy_image, dummy_mask

# =============================================
# MODEL
# =============================================

def get_deeplab_model(num_classes, dropout_rate=0.3):
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights='DeepLabV3_ResNet50_Weights.DEFAULT'
        )

        model.classifier[4] = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

        frozen_count = 0
        for name, param in model.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
                frozen_count += 1

        print(f"✅ Model loaded with dropout={dropout_rate}, frozen {frozen_count} parameters")

        if hasattr(model.backbone, 'enable_gradient_checkpointing'):
            model.backbone.enable_gradient_checkpointing = True

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# =============================================
# MIXUP
# =============================================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if isinstance(criterion, CombinedLoss):
        loss_a, comp_a = criterion(pred, y_a)
        loss_b, comp_b = criterion(pred, y_b)
        total_loss = lam * loss_a + (1 - lam) * loss_b

        components = {}
        for key in comp_a.keys():
            components[key] = lam * comp_a[key] + (1 - lam) * comp_b[key]

        return total_loss, components
    else:
        loss_a = criterion(pred, y_a)
        loss_b = criterion(pred, y_b)
        return lam * loss_a + (1 - lam) * loss_b, {}

# =============================================
# METRICS
# =============================================

def calculate_metrics(pred, target, num_classes):
    try:
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()

        precision = precision_score(target_flat, pred_flat, average='macro', zero_division=0)
        recall = recall_score(target_flat, pred_flat, average='macro', zero_division=0)
        f1 = f1_score(target_flat, pred_flat, average='macro', zero_division=0)
        acc = accuracy_score(target_flat, pred_flat)

        iou_scores = []
        pred_cpu, target_cpu = pred.view(-1).cpu(), target.view(-1).cpu()
        for cls in range(num_classes):
            pred_inds = (pred_cpu == cls)
            target_inds = (target_cpu == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union > 0:
                iou_scores.append(intersection / union)
            else:
                iou_scores.append(0.0)
        miou = np.mean(iou_scores)

        cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))

        return precision, recall, f1, acc, miou, cm

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros((num_classes, num_classes))

def create_cm_plot(cm, num_classes):
    class_names = ['Background', 'Jamur', 'Foxing', 'Noda', 'Korosi Tinta'][:num_classes]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Validation Confusion Matrix')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    pil_img = Image.open(buf)
    img = wandb.Image(pil_img)

    plt.close(fig)
    return img

# =============================================
# TRAINING & VALIDATION
# =============================================

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, use_mixup=True):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    accumulation_steps = CONFIG["gradient_accumulation_steps"]
    loss_components_sum = {}

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        try:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if use_mixup and random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=CONFIG["mixup_alpha"])
                outputs = model(images)
                logits = outputs['out']

                if isinstance(criterion, CombinedLoss):
                    loss, components = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    for key, value in components.items():
                        loss_components_sum[key] = loss_components_sum.get(key, 0.0) + value
                else:
                    loss, _ = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

                with torch.no_grad():
                    preds = torch.argmax(logits.detach(), dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(labels_a.cpu())
            else:
                outputs = model(images)
                logits = outputs['out']

                if isinstance(criterion, CombinedLoss):
                    loss, components = criterion(logits, labels)
                    for key, value in components.items():
                        loss_components_sum[key] = loss_components_sum.get(key, 0.0) + value
                else:
                    loss = criterion(logits, labels)

                with torch.no_grad():
                    preds = torch.argmax(logits.detach(), dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_loss += loss.item() * accumulation_steps

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: OOM at batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    optimizer.step()
    optimizer.zero_grad()

    if all_preds and all_targets:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        prec, rec, f1, acc, miou, _ = calculate_metrics(all_preds, all_targets, num_classes)
    else:
        prec = rec = f1 = acc = miou = 0.0

    avg_components = {key: value / len(loader) for key, value in loss_components_sum.items()}

    return total_loss / len(loader), prec, rec, f1, acc, miou, avg_components

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    loss_components_sum = {}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Validating", leave=False)):
            try:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                logits = outputs['out']

                if isinstance(criterion, CombinedLoss):
                    loss, components = criterion(logits, labels)
                    for key, value in components.items():
                        loss_components_sum[key] = loss_components_sum.get(key, 0.0) + value
                else:
                    loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: OOM at validation batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    if all_preds and all_targets:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        prec, rec, f1, acc, miou, cm = calculate_metrics(all_preds, all_targets, num_classes)
    else:
        prec = rec = f1 = acc = miou = 0.0
        cm = np.zeros((num_classes, num_classes))

    avg_components = {key: value / len(loader) for key, value in loss_components_sum.items()}

    return total_loss / len(loader), prec, rec, f1, acc, miou, cm, avg_components

# =============================================
# MAIN
# =============================================

def main():
    try:
        train_dataset = FungiDataset(
            "/content/drive/MyDrive/traindataset/dataset_masks/train/images",
            "/content/drive/MyDrive/traindataset/dataset_masks/train/label_masks",
            is_train=True
        )
        val_dataset = FungiDataset(
            "/content/drive/MyDrive/traindataset/dataset_masks/valid/images",
            "/content/drive/MyDrive/traindataset/dataset_masks/valid/label_masks",
            is_train=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG["val_batch_size"], shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False
        )

        print("Loading model...")
        model = get_deeplab_model(CONFIG["num_classes"], CONFIG["dropout_rate"]).to(CONFIG["device"])

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"]
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        print("\n" + "="*60)
        criterion = get_loss_function(CONFIG["loss_config"])
        print("="*60 + "\n")

        early_stopping = EarlyStopping(
            patience=CONFIG["early_stopping_patience"],
            min_delta=CONFIG["min_delta"],
            verbose=True
        )

        best_val_miou = -1.0

        print("Starting training with Early Stopping...")
        print(f"Early stopping patience: {CONFIG['early_stopping_patience']} epochs")

        for epoch in range(CONFIG["epochs"]):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"{'='*60}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_loss, train_prec, train_rec, train_f1, train_acc, train_miou, train_components = train_one_epoch(
                model, train_loader, optimizer, criterion, CONFIG["device"],
                CONFIG["num_classes"], use_mixup=CONFIG["use_mixup"]
            )

            val_loss, val_prec, val_rec, val_f1, val_acc, val_miou, val_cm, val_components = validate(
                model, val_loader, criterion, CONFIG["device"], CONFIG["num_classes"]
            )

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            cm_plot = create_cm_plot(val_cm, CONFIG["num_classes"])

            train_val_gap_miou = train_miou - val_miou
            train_val_gap_acc = train_acc - val_acc

            wandb_log = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train/loss": train_loss,
                "train/precision": train_prec,
                "train/recall": train_rec,
                "train/f1": train_f1,
                "train/accuracy": train_acc,
                "train/mIoU": train_miou,
                "val/loss": val_loss,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
                "val/accuracy": val_acc,
                "val/mIoU": val_miou,
                "val/confusion_matrix": cm_plot,
                "overfitting/miou_gap": train_val_gap_miou,
                "overfitting/accuracy_gap": train_val_gap_acc,
            }

            if train_components:
                for key, value in train_components.items():
                    wandb_log[f"train/loss_{key}"] = value
            if val_components:
                for key, value in val_components.items():
                    wandb_log[f"val/loss_{key}"] = value

            wandb.log(wandb_log)

            print(f"\n📊 Training Metrics:")
            print(f"   Loss: {train_loss:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f}")
            print(f"   F1: {train_f1:.4f} | Accuracy: {train_acc:.4f} | mIoU: {train_miou:.4f}")
            if train_components:
                components_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_components.items()])
                print(f"   Components: {components_str}")

            print(f"\n📊 Validation Metrics:")
            print(f"   Loss: {val_loss:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")
            print(f"   F1: {val_f1:.4f} | Accuracy: {val_acc:.4f} | mIoU: {val_miou:.4f}")
            if val_components:
                components_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_components.items()])
                print(f"   Components: {components_str}")

            print(f"\n⚠️  Overfitting Check:")
            print(f"   Train-Val mIoU Gap: {train_val_gap_miou:.4f} ({train_val_gap_miou*100:.2f}%)")

            if train_val_gap_miou > 0.25:
                print(f"   🚨 HIGH OVERFITTING DETECTED!")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                save_path = "/content/drive/MyDrive/traindataset/deeplab3_fungi_best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_miou': best_val_miou,
                    'config': CONFIG
                }, save_path)
                print(f"\n💾 Model saved! New best validation mIoU: {best_val_miou:.4f}")

            if early_stopping(val_miou, epoch + 1):
                print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
                print(f"   Best validation mIoU: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
                break

        print("\n" + "="*60)
        print("✅ Training complete!")
        print(f"   Best validation mIoU: {best_val_miou:.4f}")
        print(f"   Total epochs trained: {epoch + 1}")
        print("="*60)

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise e
    finally:
        wandb.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
