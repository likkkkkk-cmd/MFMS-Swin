import seaborn as sns
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report

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
# 1. 配置参数 (已根据截图路径修正)
# ==========================================
CONFIG = {
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
    'vis_dir': 'vis_results'
}

# ==========================================
# 2. 辅助功能
# ==========================================
def plot_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.plot(epochs, history['train_loss'], 'b-'); plt.title('Total Loss')
    plt.subplot(1, 4, 2); plt.plot(epochs, history['train_acc'], 'b-', label='Train'); plt.plot(epochs, history['val_acc'], 'r-', label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 4, 3); plt.plot(epochs, history['val_kappa'], 'g-'); plt.title('Kappa')
    plt.subplot(1, 4, 4); plt.plot(epochs, history['con_loss'], 'y-'); plt.title('Mining Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix_heatmap(true_labels, pred_labels, save_dir, filename='confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2', '3', '4'], 
                yticklabels=['0', '1', '2', '3', '4'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {filename.split(".")[0]}')
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()

def visualize_heatmap(model, loader, device, save_dir, num_images=8):
    model.eval()
    save_path = os.path.join(save_dir, CONFIG['vis_dir'])
    os.makedirs(save_path, exist_ok=True)
    data_iter = iter(loader)
    try: images, labels = next(data_iter)
    except: return
    images = images.to(device)
    with torch.no_grad():
        out = model(images, return_map=True)
        # 兼容无掩码模型的假掩码返回
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
# 3. 核心网络组件与损失函数
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
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        loss_cls = self.ce(logits, targets)
        if len(targets.shape) > 1: true_ranks = targets.argmax(dim=1).float()
        else: true_ranks = targets.float()
            
        probs = torch.softmax(logits, dim=1)
        classes = torch.arange(logits.size(1), device=logits.device).float()
        pred_ranks = torch.sum(probs * classes, dim=1)
        
        loss_ordinal = self.mse(pred_ranks, true_ranks)
        return loss_cls + 0.1 * loss_ordinal

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
# 4. 消融实验模型集合
# ==========================================
def load_swin_weights(model, pretrained_path):
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('layers.', 'layers_') if k.startswith('layers.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

# 基线模型 (仅Swin-Tiny)
class BaselineSwin(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None):
        super(BaselineSwin, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True, img_size=CONFIG['img_size'], drop_path_rate=0.2)
        load_swin_weights(self.backbone, pretrained_path)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x, return_map=False):
        features = self.backbone(x)
        feat_map = features[-1].permute(0, 3, 1, 2) 
        logits = self.head(self.avg_pool(feat_map).flatten(1))
        if return_map:
            dummy_map = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            return logits, dummy_map
        return logits

# 完整模型 (Ours)
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

    def forward(self, x, return_map=False):
        feat_map = self.extract_features(x)
        logits_1 = self.head(self.avg_pool(feat_map).flatten(1))
        att_map_small = self.att_gen(feat_map)
        att_map_large = F.interpolate(att_map_small, size=x.shape[2:], mode='bilinear', align_corners=True)
        if not self.training or return_map:
            if return_map: return logits_1, att_map_large
            return logits_1
        mask = att_map_large
        x_masked = x * mask
        feat_map_2 = self.extract_features(x_masked)
        logits_2 = self.head(self.avg_pool(feat_map_2).flatten(1))
        return logits_1, logits_2, att_map_large

# ==========================================
# 5. 数据加载与处理 (增强了兼容性)
# ==========================================
class DirectoryDataset(Dataset):
    def __init__(self, csv_file, img_dir, root_dir, transform=None):
        self.csv_path = os.path.join(root_dir, csv_file)
        self.img_dir_path = os.path.join(root_dir, img_dir)
        self.transform = transform
        try: self.df = pd.read_csv(self.csv_path)
        except: self.df = pd.read_excel(self.csv_path)

        # 动态推断列名以兼容不同格式的CSV表头
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
# 6. 训练与评估流程
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, mixup_fn=None, model_ema=None, current_epoch=0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if mixup_fn is not None: images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits_1, logits_2, att_map = outputs
                loss_cls = criterion(logits_1, labels)
                if current_epoch < 25:
                    total_loss = loss_cls
                else:
                    loss_con = criterion(logits_2, labels)
                    loss_sparsity = torch.mean(att_map)
                    dynamic_cons_weight = min(CONFIG['cons_weight'], (current_epoch - 25) / 20 * CONFIG['cons_weight'])
                    total_loss = loss_cls + dynamic_cons_weight * loss_con + 0.03 * loss_sparsity
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

    return running_loss / len(loader.dataset), 0.0, accuracy_score(all_labels, all_preds)

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
# 7. 主程序
# ==========================================
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print(f"Using device: {CONFIG['device']} | Batch Size: {CONFIG['batch_size']}")
    
    train_ds = DirectoryDataset(CONFIG['train_csv'], CONFIG['train_dir'], CONFIG['root_dir'], transform=train_transform)
    val_ds = DirectoryDataset(CONFIG['val_csv'], CONFIG['val_dir'], CONFIG['root_dir'], transform=val_test_transform)
    test_ds = DirectoryDataset(CONFIG['test_csv'], CONFIG['test_dir'], CONFIG['root_dir'], transform=val_test_transform)
    
    target_list = [label.item() for _, label in train_ds]
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 完整版 (Ours)
    model_name = "ours_full"
    model = InterpretableMFMSSwin(num_classes=CONFIG['num_classes'], pretrained_path=CONFIG['pretrained_path']).to(CONFIG['device'])
    
    # ------------------------------------------

    print(f"Initializing Model: {model_name} ...")
    model_ema = ModelEmaV2(model, decay=0.999, device=CONFIG['device'])

    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.0, prob=0.3, switch_prob=0.0, mode='batch', label_smoothing=0.1, num_classes=CONFIG['num_classes'])

    class_counts = torch.bincount(torch.tensor(target_list)).float()
    weights = 1.0 / (class_counts ** 0.25)
    weights = weights / weights.sum() * CONFIG['num_classes']
    weights = weights.to(CONFIG['device'])
    
    criterion_train = OrdinalFocalLoss(weight=weights, gamma=2.0)
    criterion_val = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    scaler = GradScaler()

    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_kappa': []}
    best_kappa = 0.0
    best_model_path = os.path.join(CONFIG['root_dir'], f'best_model_{model_name}.pth')

    print(f"Start Training {model_name}...")
    for epoch in range(CONFIG['epochs']):
        train_loss, _, train_acc = train_epoch(model, train_loader, criterion_train, optimizer, CONFIG['device'], scaler, mixup_fn, model_ema, current_epoch=epoch)
        val_loss, val_acc, val_kappa = evaluate_with_tta(model_ema.module, val_loader, criterion_val, CONFIG['device'])
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Kappa: {val_kappa:.4f}")
        
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model_ema.module.state_dict(), best_model_path)
            print(f">>> Saved Best Model (EMA) to {best_model_path}")

    # 评估与导出
    print(f"\nFinal Evaluation with Best Model ({model_name})...")
    model.load_state_dict(torch.load(best_model_path))

    _, val_acc, val_kappa, val_true, val_pred = evaluate_with_tta(model, val_loader, criterion_val, CONFIG['device'], return_details=True)
    pd.DataFrame({'id_code': val_ds.df[val_ds.img_col].values, 'true_label': val_true, 'pred_label': val_pred, 'error_distance': np.abs(np.array(val_true) - np.array(val_pred))}).to_csv(os.path.join(CONFIG['root_dir'], f'val_preds_{model_name}.csv'), index=False)
    plot_confusion_matrix_heatmap(val_true, val_pred, CONFIG['root_dir'], filename=f'val_cm_{model_name}.png')
    print(f"验证集完成。Kappa: {val_kappa:.4f} | Accuracy: {val_acc:.4f}")

    _, test_acc, test_kappa, test_true, test_pred = evaluate_with_tta(model, test_loader, criterion_val, CONFIG['device'], return_details=True)
    pd.DataFrame({'id_code': test_ds.df[test_ds.img_col].values, 'true_label': test_true, 'pred_label': test_pred, 'error_distance': np.abs(np.array(test_true) - np.array(test_pred))}).to_csv(os.path.join(CONFIG['root_dir'], f'test_preds_{model_name}.csv'), index=False)
    plot_confusion_matrix_heatmap(test_true, test_pred, CONFIG['root_dir'], filename=f'test_cm_{model_name}.png')
    print(f"测试集完成。Kappa: {test_kappa:.4f} | Accuracy: {test_acc:.4f}")

    # 1. 生成并打印验证集分类报告
    val_report = classification_report(
        val_true, val_pred, 
        target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], 
        digits=4
    )
    print("\n验证集 分类报告 (Classification Report):")
    print(val_report)
    
    with open(os.path.join(CONFIG['root_dir'], f'val_report_{model_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Validation Set Report ({model_name}):\n" + val_report)

    # 2. 生成并打印测试集分类报告
    test_report = classification_report(
        test_true, test_pred, 
        target_names=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)', 'Proliferative(4)'], 
        digits=4
    )
    print("\n测试集 分类报告 (Classification Report):")
    print(test_report)
    
    with open(os.path.join(CONFIG['root_dir'], f'test_report_{model_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Test Set Report ({model_name}):\n" + test_report)

    # 3. 可视化生成
    try:
        visualize_heatmap(model, val_loader, CONFIG['device'], CONFIG['root_dir'], num_images=8)
        plot_curves({
            'train_loss': history['train_loss'], 
            'con_loss': history['train_loss'], 
            'train_acc': history['train_acc'], 
            'val_acc': history['val_acc'], 
            'val_kappa': history['val_kappa']
        }, CONFIG['root_dir'])
    except Exception as e:
        pass