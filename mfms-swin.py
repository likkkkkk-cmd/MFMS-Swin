import seaborn as sns
import os
import copy
import random
import re

# 固定 CUDA 矩阵运算的工作空间配置；需在首次使用 CUDA 前设置。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import timm
from timm.data.mixup import Mixup
from timm.utils import ModelEmaV2

# ==========================================
# 1. 配置参数（新增消融实验配置）
# ==========================================
CONFIG = {
    'seed': 42,
    'use_cross_validation': True,  # False 时保留原 train/val/test 固定划分
    'n_folds': 5,
    'cv_dir': 'cv_results',
    'root_dir': "/root/shared-nvme/xiaolunwen/dataset/ddr/split_data",
    'train_csv': "train.csv",
    'val_csv': "val.csv",
    'test_csv': "test.csv",
    'train_dir': "train_process",
    'val_dir': "val_process",
    'test_dir': "test_process",
    'pretrained_path': "/root/shared-nvme/my_project/archive/swin_tiny_patch4_window7_224.pth",
    'img_size': 384,
    'num_classes': 5,
    'batch_size': 32,
    'epochs': 60,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cons_weight': 0.1,
    'vis_dir': 'vis_results',
    # 消融实验配置
    'ablation_models': [
        'baseline_swin',      # 基线：纯Swin-T
        'swin_with_mfms',     # 仅MFMS（无Masking）
        'swin_with_masking',  # 仅Masking（无MFMS）
        'full_model'          # 完整模型（MFMS+Masking）
    ]
}

# ==========================================
# 随机种子：主进程、CUDA、数据加载子进程和采样器
# ==========================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    # DataLoader 已设置各 worker 的 PyTorch 种子，再同步 Python/NumPy。
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_seeded_loader(dataset, seed, **kwargs):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, worker_init_fn=seed_worker, generator=generator, **kwargs)

# ==========================================
# 2. 辅助功能（复用原代码）
# ==========================================
def plot_curves(history, save_dir, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.plot(epochs, history['train_loss'], 'b-'); plt.title('Total Loss')
    plt.subplot(1, 4, 2); plt.plot(epochs, history['train_acc'], 'b-', label='Train'); plt.plot(epochs, history['val_acc'], 'r-', label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 4, 3); plt.plot(epochs, history['val_kappa'], 'g-'); plt.title('Kappa')
    if 'con_loss' in history and len(history['con_loss']) > 0:
        plt.subplot(1, 4, 4); plt.plot(epochs, history['con_loss'], 'y-'); plt.title('Mining Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{model_name}.png'))
    plt.close()

def plot_confusion_matrix_heatmap(true_labels, pred_labels, save_dir, model_name, filename='confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2', '3', '4'], 
                yticklabels=['0', '1', '2', '3', '4'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}_{filename.split(".")[0]}')
    plt.savefig(os.path.join(save_dir, f'{model_name}_{filename}'), bbox_inches='tight')
    plt.close()

def visualize_heatmap(model, loader, device, save_dir, model_name, num_images=8):
    model.eval()
    save_path = os.path.join(save_dir, f'{CONFIG["vis_dir"]}_{model_name}')
    os.makedirs(save_path, exist_ok=True)
    data_iter = iter(loader)
    try: images, labels = next(data_iter)
    except: return
    images = images.to(device)
    with torch.no_grad():
        out = model(images, return_map=True)
        preds = out[0] if isinstance(out, tuple) else out
        att_maps = out[1] if isinstance(out, tuple) else torch.ones((images.size(0), 1, images.size(2), images.size(3))).to(device)
        preds = torch.argmax(preds, dim=1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for i in range(min(num_images, len(images))):
        img_tensor = images[i] * std + mean
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)

        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        radius = int(min(h, w) / 2 * 0.96)
        center = (int(w / 2), int(h / 2))
        cv2.circle(mask, center, radius, (1.0), -1)

        att = att_maps[i, 0].cpu().numpy()
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        att = att * mask
        
        img_masked = img_np * mask[:, :, np.newaxis]
        
        heatmap = cv2.applyColorMap(np.uint8(255 * att), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        heatmap = heatmap * mask[:, :, np.newaxis]

        overlay = 0.5 * img_masked + 0.5 * heatmap
        overlay = overlay * mask[:, :, np.newaxis]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_masked); plt.title(f"Original (GT:{labels[i]})"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(att, cmap='jet'); plt.title("Attention Map"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title(f"Overlay (Pred:{preds[i]})"); plt.axis('off')
        plt.savefig(os.path.join(save_path, f"vis_sample_{i}.png"), bbox_inches='tight')
        plt.close()

# ==========================================
# 3. 核心网络组件与损失函数（复用原代码）
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if len(targets.shape) > 1:
            log_pt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(log_pt)
            focal_weight = (1 - pt) ** self.gamma
            if self.weight is not None:
                focal_weight = focal_weight * self.weight.to(inputs.device)
            loss = torch.sum(-targets * focal_weight * log_pt, dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma) * ce_loss
            
        if self.reduction == 'mean': 
            return loss.mean()
        return loss

class OrdinalFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.ce = FocalLoss(weight=weight, gamma=gamma, reduction='mean')
        self.l1 = nn.L1Loss()

    def forward(self, logits, targets):
        loss_cls = self.ce(logits, targets)
        classes = torch.arange(logits.size(1), device=logits.device).float()
        
        if len(targets.shape) > 1: 
            true_ranks = torch.sum(targets * classes, dim=1)
        else: 
            true_ranks = targets.float()
            
        probs = torch.softmax(logits, dim=1)
        pred_ranks = torch.sum(probs * classes, dim=1)
        
        loss_ordinal = self.l1(pred_ranks, true_ranks)
        return loss_cls + 0.2 * loss_ordinal

class MFMSModule(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(MFMSModule, self).__init__()
        self.conv_1x1_list = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list])
        self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(), nn.Linear(out_channels // 2, out_channels), nn.Softmax(dim=1))

    def forward(self, features):
        features = [f.permute(0, 3, 1, 2) for f in features]
        target_size = features[0].shape[2:]
        processed_feats = []
        for idx, feat in enumerate(features):
            x = self.conv_1x1_list[idx](feat)
            if x.shape[2:] != target_size: x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            processed_feats.append(x)
        concat_feat = torch.cat(processed_feats, dim=1)
        fused_feat = self.fusion_conv(concat_feat)
        spatial_attn = torch.sigmoid(self.spatial_conv(fused_feat))
        out_spatial = fused_feat * spatial_attn
        b, c, h, w = fused_feat.shape
        channel_desc = self.global_avg_pool(fused_feat).view(b, c)
        channel_weights = self.fc(channel_desc).view(b, c, 1, 1)
        out_channel = fused_feat * channel_weights * torch.sigmoid(fused_feat)
        return out_spatial + out_channel

# ==========================================
# 4. 消融实验模型集合（核心修改）
# ==========================================
def load_swin_weights(model, pretrained_path):
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('layers.', 'layers_') if k.startswith('layers.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

# 消融模型1：基线模型（纯Swin-T，无MFMS、无Masking）
class BaselineSwin(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None):
        super(BaselineSwin, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=False, img_size=CONFIG['img_size'], drop_path_rate=0.2, num_classes=num_classes)
        load_swin_weights(self.backbone, pretrained_path)
        # 兼容可视化的dummy att_gen
        self.att_gen = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, return_map=False, use_mask=True):
        logits = self.backbone(x)
        if return_map:
            # 生成dummy attention map用于可视化兼容
            att_map = torch.ones((x.size(0), 1, x.size(2), x.size(3))).to(x.device)
            return logits, att_map
        return logits

# 消融模型2：仅MFMS（Swin-T + MFMS，无Masking）
class SwinWithMFMS(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None):
        super(SwinWithMFMS, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True, img_size=CONFIG['img_size'], drop_path_rate=0.2)
        load_swin_weights(self.backbone, pretrained_path)
        feature_channels = self.backbone.feature_info.channels()
        self.mfms = MFMSModule(in_channels_list=feature_channels, out_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)
        # 兼容可视化的dummy att_gen
        self.att_gen = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        return self.mfms(self.backbone(x))

    def forward(self, x, return_map=False, use_mask=True):
        feat_map = self.extract_features(x)
        logits = self.head(self.avg_pool(feat_map).flatten(1))
        if return_map:
            att_map_small = self.att_gen(feat_map)
            att_map_large = F.interpolate(att_map_small, size=x.shape[2:], mode='bilinear', align_corners=True)
            return logits, att_map_large
        return logits

# 消融模型3：仅Masking（Swin-T + Masking，无MFMS）
class SwinWithMasking(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None):
        super(SwinWithMasking, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True, img_size=CONFIG['img_size'], drop_path_rate=0.2)
        load_swin_weights(self.backbone, pretrained_path)
        # 直接用backbone最后一层特征做attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.feature_info.channels()[-1], num_classes)
        self.att_gen = nn.Sequential(
            nn.Conv2d(self.backbone.feature_info.channels()[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        feats = self.backbone(x)
        return feats[-1]  # 取最后一层特征

    def forward(self, x, return_map=False, use_mask=True):
        feat_map = self.extract_features(x)
        logits_1 = self.head(self.avg_pool(feat_map).flatten(1))
        att_map_small = self.att_gen(feat_map)
        att_map_large = F.interpolate(att_map_small, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        if not self.training or return_map or not use_mask:
            if return_map: return logits_1, att_map_large
            return logits_1
            
        mask = att_map_large
        x_masked = x * mask
        feat_map_2 = self.extract_features(x_masked)
        logits_2 = self.head(self.avg_pool(feat_map_2).flatten(1))
        return logits_1, logits_2, att_map_large

# 消融模型4：完整模型（原代码模型）
class InterpretableMFMSSwin(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None):
        super(InterpretableMFMSSwin, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True, img_size=CONFIG['img_size'], drop_path_rate=0.2)
        load_swin_weights(self.backbone, pretrained_path)
        feature_channels = self.backbone.feature_info.channels()
        self.mfms = MFMSModule(in_channels_list=feature_channels, out_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)
        self.att_gen = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def extract_features(self, x): return self.mfms(self.backbone(x))

    def forward(self, x, return_map=False, use_mask=True):
        feat_map = self.extract_features(x)
        logits_1 = self.head(self.avg_pool(feat_map).flatten(1))
        att_map_small = self.att_gen(feat_map)
        att_map_large = F.interpolate(att_map_small, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        if not self.training or return_map or not use_mask:
            if return_map: return logits_1, att_map_large
            return logits_1
            
        mask = att_map_large
        x_masked = x * mask
        feat_map_2 = self.extract_features(x_masked)
        logits_2 = self.head(self.avg_pool(feat_map_2).flatten(1))
        return logits_1, logits_2, att_map_large

# 模型构建函数
def build_ablation_model(model_name, num_classes=5, pretrained_path=None):
    if model_name == 'baseline_swin':
        return BaselineSwin(num_classes, pretrained_path)
    elif model_name == 'swin_with_mfms':
        return SwinWithMFMS(num_classes, pretrained_path)
    elif model_name == 'swin_with_masking':
        return SwinWithMasking(num_classes, pretrained_path)
    elif model_name == 'full_model':
        return InterpretableMFMSSwin(num_classes, pretrained_path)
    else:
        raise ValueError(f"Unknown ablation model: {model_name}")

# ==========================================
# 5. 数据加载与处理（复用原代码）
# ==========================================
class DirectoryDataset(Dataset):
    def __init__(self, csv_file, img_dir, root_dir, transform=None):
        self.csv_path = os.path.join(root_dir, csv_file)
        self.img_dir_path = os.path.join(root_dir, img_dir)
        self.transform = transform
        try: self.df = pd.read_csv(self.csv_path)
        except: self.df = pd.read_excel(self.csv_path)

        self.img_col = next((col for col in ['id_code', 'image_id', 'image', 'name'] if col in self.df.columns), self.df.columns[0])
        self.label_col = next((col for col in ['diagnosis', 'dr_level', 'level', 'grade', 'label'] if col in self.df.columns), self.df.columns[1])

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        base_path = os.path.join(self.img_dir_path, img_name)
        if os.path.exists(base_path): img_path = base_path
        elif os.path.exists(base_path + '.png'): img_path = base_path + '.png'
        elif os.path.exists(base_path + '.jpg'): img_path = base_path + '.jpg'
        else: img_path = base_path
            
        try: img = Image.open(img_path).convert('RGB')
        except: img = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']))
        
        if self.transform: img = self.transform(img)
        return img, torch.tensor(int(row[self.label_col]), dtype=torch.long)

class CrossValidationDataset(Dataset):
    """复用原数据集的图像读取逻辑，每折分别应用原训练/验证变换。"""
    def __init__(self, source_datasets, records, transform=None):
        self.source_datasets = [copy.copy(dataset) for dataset in source_datasets]
        for dataset in self.source_datasets:
            dataset.transform = transform
        self.df = records.reset_index(drop=True).copy()
        self.img_col = 'id_code'
        self.label_col = 'diagnosis'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.source_datasets[int(row['source_index'])][int(row['row_index'])]


def original_image_id(image_name):
    # 与已有 huafenshuju.py 的 aug_{编号}_{原文件名} 命名方式对应。
    # 原图和增强副本共用 group_id，扩展名不参与划分。
    name = os.path.basename(str(image_name))
    original_name = re.sub(r'^(?:aug_\d+_)+', '', name)
    return os.path.splitext(original_name)[0], original_name != name


def make_cross_validation_splits(source_datasets, test_dataset, n_folds, seed):
    """合并 train/val 原图进行分层划分，保留独立测试集。"""
    if n_folds < 2:
        raise ValueError('n_folds must be at least 2.')

    rows = []
    for source_index, dataset in enumerate(source_datasets):
        for row_index in range(len(dataset)):
            row = dataset.df.iloc[row_index]
            image_name = str(row[dataset.img_col])
            group_id, is_augmented = original_image_id(image_name)
            rows.append({
                'id_code': image_name,
                'diagnosis': int(row[dataset.label_col]),
                'source_dir': dataset.img_dir_path,
                'source_index': source_index,
                'row_index': row_index,
                'group_id': group_id,
                'is_augmented': is_augmented
            })
    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError('The train/validation pool is empty.')

    originals = records.loc[~records['is_augmented']]
    if originals['group_id'].duplicated().any():
        raise ValueError('Duplicate original image IDs exist in train/val; check the CSV files.')
    if not set(records['group_id']).issubset(set(originals['group_id'])):
        raise ValueError('Some augmented images have no original image in train/val.')
    if (records.groupby('group_id')['diagnosis'].nunique() > 1).any():
        raise ValueError('An original image and its augmented copies have different labels.')

    test_groups = {
        original_image_id(name)[0]
        for name in test_dataset.df[test_dataset.img_col]
    }
    if set(records['group_id']) & test_groups:
        raise ValueError('Train/val image IDs overlap with the independent test set.')

    class_counts = originals['diagnosis'].value_counts()
    if (class_counts < n_folds).any():
        raise ValueError(f'Each class needs at least {n_folds} original images for stratified cross-validation.')

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_positions, val_positions in splitter.split(originals, originals['diagnosis']):
        train_groups = set(originals.iloc[train_positions]['group_id'])
        # 训练折保留其原图和已有增强副本；验证折仅使用原图。
        train_indices = records.index[records['group_id'].isin(train_groups)].to_numpy()
        val_indices = originals.index.to_numpy()[val_positions]
        splits.append((train_indices, val_indices))
    return records, splits


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.TrivialAugmentWide(), transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 6. 训练与评估流程（适配消融模型）
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, mixup_fn=None, model_ema=None, current_epoch=0, model_name=None):
    model.train()
    running_loss = 0.0
    running_con_loss = 0.0
    all_preds, all_labels = [], []
    
    # 仅对有masking的模型启用mask逻辑
    use_mask = (model_name in ['swin_with_masking', 'full_model']) and (current_epoch >= 10)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if mixup_fn is not None: images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images, use_mask=use_mask)
            # 处理不同模型的输出格式
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits_1, logits_2, att_map = outputs
                loss_cls = criterion(logits_1, labels)
                loss_con = criterion(logits_2, labels)
                loss_sparsity = torch.mean(att_map)
                
                progress = min(1.0, (current_epoch - 10) / 30.0)
                dynamic_cons_weight = CONFIG['cons_weight'] * progress
                dynamic_spars_weight = 0.03 * progress
                    
                total_loss = loss_cls + dynamic_cons_weight * loss_con + dynamic_spars_weight * loss_sparsity
                running_con_loss += loss_con.item() * images.size(0)
            else:
                logits_1 = outputs
                total_loss = criterion(logits_1, labels)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None: model_ema.update(model)

        running_loss += total_loss.item() * images.size(0)
        if len(labels.shape) > 1: true_labels = labels.argmax(dim=1)
        else: true_labels = labels
        _, preds = torch.max(logits_1, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(true_labels.cpu().numpy())

    return running_loss / len(loader.dataset), running_con_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)

def evaluate_with_tta(model, loader, criterion, device, return_details=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            img_flip_h = torch.flip(images, [3])
            img_flip_v = torch.flip(images, [2])
            img_rot = torch.rot90(images, 1, [2, 3])

            def get_probs(img):
                out = model(img)
                logits = out[0] if isinstance(out, tuple) else out
                return torch.softmax(logits, dim=1)

            probs_avg = (get_probs(images) + get_probs(img_flip_h) + get_probs(img_flip_v) + get_probs(img_rot)) / 4.0
            _, preds = torch.max(probs_avg, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    if return_details: return 0.0, acc, kappa, all_labels, all_preds
    return 0.0, acc, kappa

# ==========================================
# 7. 消融实验主程序
# ==========================================
def run_ablation_experiment(train_ds=None, val_ds=None, test_ds=None, output_dir=None, seed=None):
    seed = CONFIG['seed'] if seed is None else seed
    seed_everything(seed)
    output_dir = CONFIG['root_dir'] if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # 加载数据集（所有模型共用）
    if train_ds is None:
        train_ds = DirectoryDataset(CONFIG['train_csv'], CONFIG['train_dir'], CONFIG['root_dir'], transform=train_transform)
    if val_ds is None:
        val_ds = DirectoryDataset(CONFIG['val_csv'], CONFIG['val_dir'], CONFIG['root_dir'], transform=val_test_transform)
    if test_ds is None:
        test_ds = DirectoryDataset(CONFIG['test_csv'], CONFIG['test_dir'], CONFIG['root_dir'], transform=val_test_transform)
    
    target_list = [label.item() for _, label in train_ds]
    
    # 预计算类别权重（所有模型共用）
    class_counts = torch.bincount(torch.tensor(target_list), minlength=CONFIG['num_classes']).float()
    weights = 1.0 / ((class_counts ** 0.5) + 1e-6)
    weights = weights / weights.sum() * CONFIG['num_classes']
    weights = weights.to(CONFIG['device'])
    
    # 消融实验结果汇总
    ablation_results = []

    # 遍历所有消融模型
    for model_name in CONFIG['ablation_models']:
        # 同一折的消融模型使用相同种子和独立的数据加载器。
        seed_everything(seed)
        train_loader = make_seeded_loader(train_ds, seed, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
        val_loader = make_seeded_loader(val_ds, seed + 1, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = make_seeded_loader(test_ds, seed + 2, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        print("="*80)
        print(f"Training Ablation Model: {model_name}")
        print("="*80)
        
        # 构建模型
        model = build_ablation_model(model_name, num_classes=CONFIG['num_classes'], pretrained_path=CONFIG['pretrained_path']).to(CONFIG['device'])
        model_ema = ModelEmaV2(model, decay=0.999, device=CONFIG['device'])

        # 混合增强
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.0, prob=0.3, switch_prob=0.0, mode='batch', label_smoothing=0.1, num_classes=CONFIG['num_classes'])

        # 损失函数与优化器
        criterion_train = OrdinalFocalLoss(weight=weights, gamma=2.0)
        criterion_val = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
        scaler = GradScaler()

        # 训练历史
        history = {'train_loss': [], 'con_loss': [], 'train_acc': [], 'val_acc': [], 'val_kappa': []}
        best_kappa = 0.0
        best_model_path = os.path.join(output_dir, f'best_model_{model_name}.pth')

        # 训练模型
        print(f"Start Training {model_name}...")
        for epoch in range(CONFIG['epochs']):
            train_loss, train_con_loss, train_acc = train_epoch(
                model, train_loader, criterion_train, optimizer, CONFIG['device'], 
                scaler, mixup_fn, model_ema, current_epoch=epoch, model_name=model_name
            )
            val_loss, val_acc, val_kappa = evaluate_with_tta(model_ema.module, val_loader, criterion_val, CONFIG['device'])
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['con_loss'].append(train_con_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_kappa'].append(val_kappa)
            
            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | ConLoss: {train_con_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Kappa: {val_kappa:.4f}")
            
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                torch.save(model_ema.module.state_dict(), best_model_path)
                print(f">>> Saved Best Model (EMA) to {best_model_path}")

        # 评估最佳模型
        print(f"\nFinal Evaluation with Best Model ({model_name})...")
        model.load_state_dict(torch.load(best_model_path))

        # 验证集评估
        _, val_acc, val_kappa, val_true, val_pred = evaluate_with_tta(model, val_loader, criterion_val, CONFIG['device'], return_details=True)
        pd.DataFrame({
            'id_code': val_ds.df[val_ds.img_col].values, 
            'true_label': val_true, 
            'pred_label': val_pred, 
            'error_distance': np.abs(np.array(val_true) - np.array(val_pred))
        }).to_csv(os.path.join(output_dir, f'val_preds_{model_name}.csv'), index=False)
        plot_confusion_matrix_heatmap(val_true, val_pred, output_dir, model_name, filename='val_confusion_matrix.png')
        
        # 测试集评估
        _, test_acc, test_kappa, test_true, test_pred = evaluate_with_tta(model, test_loader, criterion_val, CONFIG['device'], return_details=True)
        pd.DataFrame({
            'id_code': test_ds.df[test_ds.img_col].values, 
            'true_label': test_true, 
            'pred_label': test_pred, 
            'error_distance': np.abs(np.array(test_true) - np.array(test_pred))
        }).to_csv(os.path.join(output_dir, f'test_preds_{model_name}.csv'), index=False)
        plot_confusion_matrix_heatmap(test_true, test_pred, output_dir, model_name, filename='test_confusion_matrix.png')
        
        # 分类报告
        val_report = classification_report(
            val_true, val_pred, 
            target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], 
            digits=4, output_dict=True
        )
        test_report = classification_report(
            test_true, test_pred, 
            target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], 
            digits=4, output_dict=True
        )
        
        # 保存分类报告
        with open(os.path.join(output_dir, f'val_report_{model_name}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Validation Set Report ({model_name}):\n" + classification_report(
                val_true, val_pred, target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], digits=4
            ))
        with open(os.path.join(output_dir, f'test_report_{model_name}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Test Set Report ({model_name}):\n" + classification_report(
                test_true, test_pred, target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], digits=4
            ))
        
        # 可视化
        try:
            visualize_heatmap(model, val_loader, CONFIG['device'], output_dir, model_name, num_images=8)
            plot_curves(history, output_dir, model_name)
        except Exception as e:
            print(f"Visualization error for {model_name}: {e}")
        
        # 记录消融实验结果
        ablation_results.append({
            'model_name': model_name,
            'val_acc': val_acc,
            'val_kappa': val_kappa,
            'test_acc': test_acc,
            'test_kappa': test_kappa,
            'val_f1_macro': val_report['macro avg']['f1-score'],
            'test_f1_macro': test_report['macro avg']['f1-score']
        })
        
        # 清空显存
        del model, model_ema, optimizer, scheduler, scaler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 保存消融实验汇总结果
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(output_dir, 'ablation_experiment_results.csv'), index=False)
    print("\n" + "="*80)
    print("Ablation Experiment Results Summary:")
    print(ablation_df)
    return ablation_df


def run_cross_validation():
    seed_everything(CONFIG['seed'])
    # 仅重新划分原 train/val 数据；test.csv 始终保持独立。
    source_datasets = [
        DirectoryDataset(CONFIG['train_csv'], CONFIG['train_dir'], CONFIG['root_dir']),
        DirectoryDataset(CONFIG['val_csv'], CONFIG['val_dir'], CONFIG['root_dir'])
    ]
    test_ds = DirectoryDataset(CONFIG['test_csv'], CONFIG['test_dir'], CONFIG['root_dir'], transform=val_test_transform)
    records, splits = make_cross_validation_splits(source_datasets, test_ds, CONFIG['n_folds'], CONFIG['seed'])
    cv_output_dir = os.path.join(CONFIG['root_dir'], CONFIG['cv_dir'])
    os.makedirs(cv_output_dir, exist_ok=True)
    all_results = []

    for fold_number, (train_indices, val_indices) in enumerate(splits, start=1):
        fold_seed = CONFIG['seed'] + fold_number - 1
        fold_dir = os.path.join(cv_output_dir, f'fold_{fold_number:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        train_records = records.iloc[train_indices]
        val_records = records.iloc[val_indices]
        train_records.to_csv(os.path.join(fold_dir, 'train_split.csv'), index=False)
        val_records.to_csv(os.path.join(fold_dir, 'val_split.csv'), index=False)
        train_ds = CrossValidationDataset(source_datasets, train_records, transform=train_transform)
        val_ds = CrossValidationDataset(source_datasets, val_records, transform=val_test_transform)
        print(f'\nCross-validation fold {fold_number}/{CONFIG["n_folds"]} | Seed: {fold_seed} | Train: {len(train_ds)} | Validation: {len(val_ds)}')
        fold_results = run_ablation_experiment(train_ds, val_ds, test_ds, output_dir=fold_dir, seed=fold_seed)
        fold_results.insert(1, 'fold', fold_number)
        fold_results.insert(2, 'seed', fold_seed)
        all_results.append(fold_results)
        # 每完成一折保存结果，长时间训练中也可查看已有结果。
        pd.concat(all_results, ignore_index=True).to_csv(os.path.join(cv_output_dir, 'cross_validation_results.csv'), index=False)

    results = pd.concat(all_results, ignore_index=True)
    metric_columns = ['val_acc', 'val_kappa', 'test_acc', 'test_kappa', 'val_f1_macro', 'test_f1_macro']
    summary = results.groupby('model_name', sort=False)[metric_columns].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{statistic}' for metric, statistic in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, 'n_folds', CONFIG['n_folds'])
    summary.to_csv(os.path.join(cv_output_dir, 'cross_validation_summary.csv'), index=False)
    print('\nCross-Validation Results (mean and sample standard deviation):')
    print(summary)
    return summary

if __name__ == '__main__':
    if CONFIG['use_cross_validation']:
        run_cross_validation()
    else:
        run_ablation_experiment()
