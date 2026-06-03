"""
CNN宠物品种识别模型训练模块
基于 ResNet-18 迁移学习，在 Oxford-IIIT Pet 数据集上微调
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time

# ---------- 超参数配置 ----------
CONFIG = {
    "data_dir": "data/oxford_pet",
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 0.001,
    "num_classes": 37,
    "img_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "weights/pet_breed_model.pth"
}


def build_model(num_classes=37, pretrained=True):
    """构建 ResNet-18 迁移学习模型"""
    model = models.resnet18(weights="DEFAULT" if pretrained else None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def get_data_transforms():
    """定义数据增强策略"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_data(data_dir, train_transform, val_transform):
    """加载数据集，按80%/20%划分训练集与验证集"""
    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"训练集: {train_size} 张 | 验证集: {val_size} 张 | 类别数: {len(full_dataset.classes)}")
    return train_loader, val_loader, full_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    """单轮训练"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    """验证集评估"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100. * correct / total, all_preds, all_labels


def train():
    """主训练函数"""
    print(f"使用设备: {CONFIG['device']}")
    print("=" * 50)
    print("[1/4] 加载数据集...")
    train_tf, val_tf = get_data_transforms()
    train_loader, val_loader, class_names = load_data(CONFIG["data_dir"], train_tf, val_tf)
    print("[2/4] 构建模型...")
    model = build_model(num_classes=len(class_names)).to(CONFIG["device"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=CONFIG["learning_rate"])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    print("[3/4] 开始训练...")
    best_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, CONFIG["device"])
        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d}/{CONFIG['num_epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | 耗时: {elapsed:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names, "config": CONFIG, "best_acc": best_acc}, CONFIG["save_path"])
            print(f"  >>> 保存最佳模型 (验证准确率: {best_acc:.2f}%)")
    print(f"\n[4/4] 训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"模型已保存至: {CONFIG['save_path']}")


if __name__ == "__main__":
    train()
