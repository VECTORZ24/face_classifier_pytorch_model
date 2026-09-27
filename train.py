"""
train.py
Fine-tunes a pretrained ResNet18 to classify photos into:
    0: aziz
    1: ghaith
    2: unidentified
(class indices are assigned alphabetically by ImageFolder, printed at runtime)

Expected folder layout:
data/
├── train/
│   ├── ghaith/
│   ├── aziz/
│   └── unidentified/
└── val/
    ├── ghaith/
    ├── aziz/
    └── unidentified/

Usage:
    python train.py --data_dir data --epochs 15 --batch_size 16
"""

import argparse
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size, image_size=224):
    # ImageNet normalization stats, since we're using an ImageNet-pretrained backbone
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze everything except the final classifier layer.
        # With a small dataset this avoids overfitting and trains fast.
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # this layer is always trainable
    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze", action="store_true",
                         help="Fine-tune the whole backbone instead of just the final layer "
                              "(only do this if you have a decent amount of data, ~150+ imgs/class)")
    parser.add_argument("--out", type=str, default="face_classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Classes (index order): {classes}")
    dataloaders = {"train": train_loader, "val": val_loader}

    model = build_model(num_classes=len(classes), freeze_backbone=not args.unfreeze)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start = time.time()
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, args.epochs)
    print(f"Training took {(time.time() - start):.1f}s")

    # Save both the weights and the class-index mapping so test.py doesn't need to guess it
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
    }, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()