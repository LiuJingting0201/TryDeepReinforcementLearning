import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import json
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import glob
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练


class AffordanceDataset(Dataset):
    def __init__(self, data_dir, is_train=True, oversample_positives=True):
        self.data_dir = os.path.join(data_dir, 'train' if is_train else 'test')
        self.rgb_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_rgb.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_depth.npy')))
        self.afford_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_affordance.npy')))
        self.angle_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_angles.npy')))
        self.transform = transforms.ToTensor()
        self.num_angle_classes = 36  # 每10度一类，0-35对应0°-350°

        # 过采样正样本场景
        if is_train and oversample_positives:
            self._oversample_positive_scenes()

    def _oversample_positive_scenes(self):
        """过采样包含正affordance像素的场景"""
        original_indices = list(range(len(self.rgb_paths)))
        positive_indices = []

        # 找到包含正样本的场景
        for idx in original_indices:
            afford = np.load(self.afford_paths[idx])
            if np.sum(afford > 0.5) > 0:  # 包含正样本
                positive_indices.append(idx)

        # 过采样正样本场景 (复制3倍)
        oversampled_indices = original_indices + positive_indices * 3

        # 更新路径列表
        self.rgb_paths = [self.rgb_paths[i] for i in oversampled_indices]
        self.depth_paths = [self.depth_paths[i] for i in oversampled_indices]
        self.afford_paths = [self.afford_paths[i] for i in oversampled_indices]
        self.angle_paths = [self.angle_paths[i] for i in oversampled_indices]

        print(f"   过采样后训练场景: {len(self.rgb_paths)} (原始: {len(original_indices)}, 正样本场景: {len(positive_indices)})")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = self.transform(rgb)  # (3,H,W)

        depth = np.load(self.depth_paths[idx])
        depth = torch.tensor(depth).unsqueeze(0).float()  # (1,H,W)

        afford = np.load(self.afford_paths[idx])
        afford = torch.tensor(afford).unsqueeze(0).float()  # (1,H,W)

        angle = np.load(self.angle_paths[idx])
        # 离散化角度：0-360度 -> 0-35类
        angle_class = torch.tensor((angle / 10).astype(int) % self.num_angle_classes).long()  # (H,W)

        # 只对 affordance=1 的像素计算角度损失
        valid_angle_mask = (afford.squeeze(0) > 0.5).float()

        x = torch.cat([rgb, depth], dim=0)  # (4,H,W)
        return x, afford.squeeze(0), angle_class.squeeze(0), valid_angle_mask.squeeze(0)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=37):  # 36角度类 + 1 affordance
        super(UNet, self).__init__()
        # 编码器 (Encoder) - 特征提取
        self.enc1 = self.conv_block(in_channels, 64)    # 4 → 64
        self.enc2 = self.conv_block(64, 128)            # 64 → 128
        self.enc3 = self.conv_block(128, 256)           # 128 → 256
        self.enc4 = self.conv_block(256, 512)           # 256 → 512

        self.pool = nn.MaxPool2d(2)  # 2x2池化，下采样

        # 解码器 (Decoder) - 特征重建
        self.dec3 = self.conv_block(512, 256)           # 512 → 256
        self.dec2 = self.conv_block(256, 128)           # 256 → 128
        self.dec1 = self.conv_block(128, 64)            # 128 → 64

        # 上采样 (Upsampling)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 512 → 256
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 256 → 128
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 128 → 64

        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, 1)     # 64 → 37

    def conv_block(self, in_c, out_c):
        """卷积块: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),  # 3x3卷积，保持尺寸
            nn.BatchNorm2d(out_c),                  # 批归一化
            nn.ReLU(inplace=True),                  # 激活函数
            nn.Conv2d(out_c, out_c, 3, padding=1),  # 第二个3x3卷积
            nn.BatchNorm2d(out_c),                  # 批归一化
            nn.ReLU(inplace=True)                   # 激活函数
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.upconv3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

class UNetLarge(nn.Module):
    """更深更大的UNet，适合300场景训练，5层架构"""
    def __init__(self, in_channels=4, out_channels=37):
        super(UNetLarge, self).__init__()
        # 更宽更深的通道 (1.5x深度)
        self.enc1 = self.conv_block(in_channels, 96)    # 4 → 96
        self.enc2 = self.conv_block(96, 192)            # 96 → 192
        self.enc3 = self.conv_block(192, 384)           # 192 → 384
        self.enc4 = self.conv_block(384, 768)           # 384 → 768
        self.enc5 = self.conv_block(768, 1536)          # 768 → 1536 (新增第5层)

        self.pool = nn.MaxPool2d(2)

        # 对应的解码器
        self.dec4 = self.conv_block(1536, 768)          # 1536 → 768
        self.dec3 = self.conv_block(768, 384)           # 768 → 384
        self.dec2 = self.conv_block(384, 192)           # 384 → 192
        self.dec1 = self.conv_block(192, 96)            # 192 → 96

        self.upconv4 = nn.ConvTranspose2d(1536, 768, 2, stride=2)  # 1536 → 768
        self.upconv3 = nn.ConvTranspose2d(768, 384, 2, stride=2)   # 768 → 384
        self.upconv2 = nn.ConvTranspose2d(384, 192, 2, stride=2)   # 384 → 192
        self.upconv1 = nn.ConvTranspose2d(192, 96, 2, stride=2)    # 192 → 96

        self.final = nn.Conv2d(96, out_channels, 1)

        # Dropout for regularization (更强的正则化)
        self.dropout = nn.Dropout2d(0.15)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with 5 levels (更深)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))  # 新增第5层

        # Decoder with 5 levels (对应解码)
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)  # 更强的dropout

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


def calculate_metrics(pred_afford, true_afford, pred_angle, true_angle, valid_mask):
    """计算训练指标"""
    metrics = {}

    # Affordance metrics
    pred_afford_prob = torch.sigmoid(pred_afford)
    pred_afford_binary = (pred_afford_prob > 0.5).float()

    # Accuracy
    afford_acc = (pred_afford_binary == true_afford).float().mean()

    # Precision, Recall, F1 for positive class
    true_pos = (pred_afford_binary * true_afford).sum()
    pred_pos = pred_afford_binary.sum()
    actual_pos = true_afford.sum()

    precision = true_pos / (pred_pos + 1e-6)
    recall = true_pos / (actual_pos + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    metrics['afford_acc'] = afford_acc.item()
    metrics['afford_precision'] = precision.item()
    metrics['afford_recall'] = recall.item()
    metrics['afford_f1'] = f1.item()

    # Angle metrics (只对有效像素)
    if valid_mask.sum() > 0:
        angle_pred_classes = pred_angle.argmax(dim=1)
        angle_acc = (angle_pred_classes == true_angle).float()
        angle_acc = (angle_acc * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        metrics['angle_acc'] = angle_acc.item()
    else:
        metrics['angle_acc'] = 0.0

    return metrics


def train_model():
    data_dir = './data/affordance_v5'
    full_dataset = AffordanceDataset(data_dir, is_train=True)

    # 300场景: 90%训练，10%验证
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # 更小batch_size + 梯度累积 (防止OOM，模拟更大batch)
    batch_size = 4  # 从8降到4，防止5层深网络OOM
    accumulation_steps = 2  # 梯度累积2步，相当于有效batch_size=8
    effective_batch_size = batch_size * accumulation_steps

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 使用更大模型
    model = UNetLarge().cuda()  # 更大容量模型

    # 加权Focal Loss + Dice Loss 处理极度不平衡的affordance标签
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=1.0, pos_weight=50.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.pos_weight = pos_weight

        def forward(self, inputs, targets):
            bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)

            # 给正样本更高的权重来处理极度不平衡
            weights = torch.where(targets == 1, self.pos_weight, 1.0)

            focal_loss = self.alpha * weights * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()

    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            inputs = torch.sigmoid(inputs)

            # Flatten
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

            return 1 - dice

    # 组合损失: Focal Loss + Dice Loss
    criterion_afford_focal = FocalLoss(alpha=0.25, gamma=1.0, pos_weight=50.0)
    criterion_afford_dice = DiceLoss()
    criterion_angle = nn.CrossEntropyLoss(reduction='none')

    # 更激进的优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)  # 更长的周期

    # 混合精度训练设置
    scaler = GradScaler()
    # 梯度裁剪，防止梯度爆炸
    max_grad_norm = 1.0

    best_val_loss = float('inf')
    patience = 25  # 更长的耐心 (数据多，需要更多训练)
    patience_counter = 0

    print("🚀 开始训练大规模可供性模型 (300场景)...")
    print(f"   训练集: {len(train_dataset)} 个场景")
    print(f"   验证集: {len(val_dataset)} 个场景")
    print(f"   模型: UNetLarge (5层, 4→96→192→384→768→1536 channels, 增强容量)")
    print(f"   优化器: AdamW (lr={optimizer.param_groups[0]['lr']:.1e}, wd=1e-4)")
    print(f"   损失: 加权Focal+Dice Loss (afford, 正样本50x权重) + Masked CE (angle)")
    print(f"   批大小: {batch_size} (梯度累积{accumulation_steps}步, 有效batch={effective_batch_size})")
    print(f"   混合精度: AMP启用，梯度裁剪={max_grad_norm}")
    print("=" * 80)

    for epoch in range(100):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_metrics = {'afford_acc': 0, 'afford_precision': 0, 'afford_recall': 0, 'afford_f1': 0, 'angle_acc': 0}

        for batch_idx, (x, afford, angle, valid_mask) in enumerate(train_loader):
            x, afford, angle, valid_mask = x.cuda(), afford.cuda(), angle.cuda(), valid_mask.cuda()

            # 混合精度前向传播
            with autocast():
                pred = model(x)
                pred_afford = pred[:, 0]    # (B, H, W)
                pred_angle = pred[:, 1:]    # (B, 36, H, W)

                # Affordance loss: Focal Loss + Dice Loss 组合
                loss_afford_focal = criterion_afford_focal(pred_afford, afford)
                loss_afford_dice = criterion_afford_dice(pred_afford, afford)
                loss_afford = 0.7 * loss_afford_focal + 0.3 * loss_afford_dice  # 70% Focal + 30% Dice

                # Angle loss (只对 affordance=1 的像素计算)
                if valid_mask.sum() > 0:
                    angle_loss_per_pixel = criterion_angle(pred_angle, angle)  # (B, H, W)
                    # 应用mask: 只在有效像素上计算损失
                    masked_angle_loss = angle_loss_per_pixel * valid_mask  # (B, H, W)
                    loss_angle = masked_angle_loss.sum() / (valid_mask.sum() + 1e-6)  # 归一化
                else:
                    loss_angle = torch.tensor(0.0).cuda()

                # 多任务损失: 大数据下调整权重
                loss = 5.0 * loss_afford + 1.0 * loss_angle  # 大幅提高affordance权重

            # 梯度累积: 每accumulation_steps步才更新参数
            loss = loss / accumulation_steps  # 损失按累积步数缩放
            scaler.scale(loss).backward()

            # 每accumulation_steps步更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps  # 恢复原始损失值用于记录

            # 累积指标 (只在参数更新时计算，避免重复计算)
            if (batch_idx + 1) % accumulation_steps == 0:
                # 累积指标
                batch_metrics = calculate_metrics(pred_afford, afford, pred_angle, angle, valid_mask)
                for k, v in batch_metrics.items():
                    train_metrics[k] += v

        # 平均训练指标 (考虑梯度累积)
        num_effective_batches = len(train_loader) // accumulation_steps
        for k in train_metrics:
            train_metrics[k] /= max(1, num_effective_batches)  # 避免除零
        train_loss /= max(1, num_effective_batches)

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_metrics = {'afford_acc': 0, 'afford_precision': 0, 'afford_recall': 0, 'afford_f1': 0, 'angle_acc': 0}

        with torch.no_grad():
            for x, afford, angle, valid_mask in val_loader:
                x, afford, angle, valid_mask = x.cuda(), afford.cuda(), angle.cuda(), valid_mask.cuda()
                pred = model(x)
                pred_afford = pred[:, 0]
                pred_angle = pred[:, 1:]

                loss_afford_focal = criterion_afford_focal(pred_afford, afford)
                loss_afford_dice = criterion_afford_dice(pred_afford, afford)
                loss_afford = 0.7 * loss_afford_focal + 0.3 * loss_afford_dice

                if valid_mask.sum() > 0:
                    angle_loss_per_pixel = criterion_angle(pred_angle, angle)
                    masked_angle_loss = angle_loss_per_pixel * valid_mask
                    loss_angle = masked_angle_loss.sum() / (valid_mask.sum() + 1e-6)
                else:
                    loss_angle = torch.tensor(0.0).cuda()

                loss = 5.0 * loss_afford + 1.0 * loss_angle
                val_loss += loss.item()

                batch_metrics = calculate_metrics(pred_afford, afford, pred_angle, angle, valid_mask)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v

        # 平均验证指标
        num_val_batches = len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= num_val_batches
        val_loss /= num_val_batches

        # 学习率调度
        scheduler.step()

        # 日志输出
        print(f"[Epoch {epoch+1:3d}] LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {train_loss:.4f} (Afford F1: {train_metrics['afford_f1']:.3f}, Angle Acc: {train_metrics['angle_acc']:.3f}) | "
              f"Val Loss: {val_loss:.4f} (Afford F1: {val_metrics['afford_f1']:.3f}, Angle Acc: {val_metrics['angle_acc']:.3f})")

        # ===== 模型保存与早停 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('./models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, './models/affordance_model_best.pth')
            print("  💾 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  🛑 早停: {patience} 个epoch验证损失无改善")
                break

    # ===== 最终评估 =====
    print("\n📊 训练完成！加载最佳模型进行最终评估...")
    checkpoint = torch.load('./models/affordance_model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    final_metrics = {'afford_acc': 0, 'afford_precision': 0, 'afford_recall': 0, 'afford_f1': 0, 'angle_acc': 0}
    with torch.no_grad():
        for x, afford, angle, valid_mask in val_loader:
            x, afford, angle, valid_mask = x.cuda(), afford.cuda(), angle.cuda(), valid_mask.cuda()
            pred = model(x)
            pred_afford = pred[:, 0]
            pred_angle = pred[:, 1:]

            batch_metrics = calculate_metrics(pred_afford, afford, pred_angle, angle, valid_mask)
            for k, v in batch_metrics.items():
                final_metrics[k] += v

    for k in final_metrics:
        final_metrics[k] /= len(val_loader)

    print("=" * 70)
    print("🎯 最终验证集性能:")
    print(f"Afford Acc: {final_metrics['afford_acc']:.3f}")
    print(f"Afford Precision: {final_metrics['afford_precision']:.3f}")
    print(f"Afford Recall: {final_metrics['afford_recall']:.3f}")
    print(f"Afford F1: {final_metrics['afford_f1']:.3f}")
    print(f"Angle Acc: {final_metrics['angle_acc']:.3f}")
    print("=" * 70)
    print("💡 方法论关键点:")
    print("   • 自监督数据生成: 物理仿真提供标签")
    print("   • Focal Loss: 处理极度不平衡的affordance标签")
    print("   • Masked Angle Loss: 只在可抓取像素上计算角度损失")
    print("   • 多任务学习: affordance(3x权重) + angle(0.8x权重)")
    print("   • 余弦退火调度: 更好的收敛和泛化")


if __name__ == '__main__':
    train_model()
