#!/usr/bin/env python3
"""
Complete execution of Assignment4-2 notebook
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import random
import copy
from tqdm import tqdm

print("="*80)
print("ASSIGNMENT 4-2: GERMAN TRAFFIC SIGN RECOGNITION")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print("="*80)

# ============================================================================
# PART 1b: Calculate mean and standard deviation of training images
# ============================================================================
print("\n" + "="*80)
print("PART 1b: Calculating dataset mean and standard deviation...")
print("="*80)

def calculate_mean_std(data_dir, classes):
    """
    Calculate mean and std deviation for the training dataset
    """
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    for class_id in classes:
        class_path = os.path.join(data_dir, str(class_id))
        if not os.path.exists(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]

                pixel_sum += img_array.sum(axis=(0, 1))
                pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
                pixel_count += img_array.shape[0] * img_array.shape[1]
            except:
                continue

    mean = pixel_sum / pixel_count
    std_dev = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    return mean, std_dev

train_data_path = './data/Train'
test_data_path = './data/Test'
classes = [0, 1, 2, 3, 4, 5]

if os.path.exists(train_data_path):
    mean, std_dev = calculate_mean_std(train_data_path, classes)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
else:
    print("Using ImageNet statistics as placeholder:")
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

# ============================================================================
# PART 1c: Implement custom dataset class
# ============================================================================
print("\n" + "="*80)
print("PART 1c: Implementing custom dataset class...")
print("="*80)

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, training=True, transform=None, mean=None, std_dev=None):
        """
        Custom Dataset for German Traffic Sign Recognition
        """
        self.root_dir = root_dir
        self.training = training
        self.transform = transform
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406])
        self.std_dev = std_dev if std_dev is not None else np.array([0.229, 0.224, 0.225])

        # Store image paths and labels
        self.image_paths = []
        self.labels = []

        # Load data for classes 0-5
        for class_id in range(6):
            class_path = os.path.join(root_dir, str(class_id))
            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_id)

        # Define transforms based on training/test mode
        if self.training:
            self.transforms_list = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std_dev)
            ])
        else:
            self.transforms_list = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std_dev)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transforms_list(image)
        label = self.labels[idx]
        return image, label

print("✓ Custom dataset class implemented")

# ============================================================================
# PART 1d: Instantiate training and test datasets
# ============================================================================
print("\n" + "="*80)
print("PART 1d: Creating datasets...")
print("="*80)

full_train_dataset = TrafficSignDataset(
    root_dir=train_data_path,
    training=True,
    mean=mean,
    std_dev=std_dev
)

test_dataset = TrafficSignDataset(
    root_dir=test_data_path,
    training=False,
    mean=mean,
    std_dev=std_dev
)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# ============================================================================
# PART 1e: Partition training dataset and create dataloaders
# ============================================================================
print("\n" + "="*80)
print("PART 1e: Creating train/val split and dataloaders...")
print("="*80)

total_size = len(full_train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# ============================================================================
# PART 1e: Visualize two random training images
# ============================================================================
print("\n" + "="*80)
print("Visualizing random training images...")
print("="*80)

indices = random.sample(range(len(train_dataset)), 2)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for idx, ax in zip(indices, axes):
    image, label = train_dataset[idx]
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = image_np * std_dev + mean
    image_np = np.clip(image_np, 0, 1)
    ax.imshow(image_np)
    ax.set_title(f'Class: {label}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('training_samples.png')
print("✓ Saved training samples to training_samples.png")

# ============================================================================
# Q1a: Build CNN Network
# ============================================================================
print("\n" + "="*80)
print("Q1a: Building CNN Network...")
print("="*80)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(64 * 8 * 8, 6)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

print("✓ CNN architecture defined")

# ============================================================================
# Q1b: Create model
# ============================================================================
print("\n" + "="*80)
print("Q1b: Creating CNN model...")
print("="*80)

cnn_model = CNN()
print("✓ CNN model created")

# ============================================================================
# Q1c: Test forward propagation
# ============================================================================
print("\n" + "="*80)
print("Q1c: Testing forward propagation...")
print("="*80)

data_batch, labels = next(iter(train_loader))
output = cnn_model(data_batch)
print(f"Output shape: {output.shape}")
print(f"Expected shape: torch.Size([{batch_size}, 6])")
assert output.shape == torch.Size([batch_size, 6]), "Output shape mismatch!"
print("✓ Forward propagation test passed")

# ============================================================================
# Q1d: Setup loss and optimizer
# ============================================================================
print("\n" + "="*80)
print("Q1d: Setting up loss function...")
print("="*80)

loss_fn = nn.CrossEntropyLoss()
print("✓ Loss function configured")

# ============================================================================
# Q2a: Hyperparameter Grid Search
# ============================================================================
print("\n" + "="*80)
print("Q2a: Starting Hyperparameter Grid Search...")
print("="*80)

learning_rates = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
momentums = [0.85, 0.9, 0.95, 0.99]
weight_decays = [0, 1e-4, 1e-2]

grid_search_results = []
grid_search_losses = {}

total_combinations = len(learning_rates) * len(momentums) * len(weight_decays)
print(f"Total combinations: {total_combinations}")
print("Running 2 epochs per configuration...")

combination_num = 0
for lr_val in learning_rates:
    for momentum_val in momentums:
        for wd_val in weight_decays:
            combination_num += 1
            print(f"\n[{combination_num}/{total_combinations}] lr={lr_val}, momentum={momentum_val}, weight_decay={wd_val}")

            model = CNN()
            optimizer = optim.SGD(model.parameters(), lr=lr_val, momentum=momentum_val, weight_decay=wd_val)

            epoch_losses = []
            for epoch in range(2):
                model.train()
                running_loss = 0.0
                num_batches = 0

                pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/2", leave=False)
                for batch in pbar:
                    optimizer.zero_grad()
                    data, target_labels = batch

                    prediction = model(data)
                    loss = loss_fn(prediction, target_labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss = running_loss / num_batches
                epoch_losses.append(avg_loss)
                print(f"  Epoch {epoch+1}/2, Loss: {avg_loss:.4f}")

            config_key = f"lr={lr_val}_m={momentum_val}_wd={wd_val}"
            grid_search_losses[config_key] = epoch_losses
            grid_search_results.append({
                'lr': lr_val,
                'momentum': momentum_val,
                'weight_decay': wd_val,
                'final_loss': epoch_losses[-1],
                'losses': epoch_losses
            })

best_config = min(grid_search_results, key=lambda x: x['final_loss'])
print("\n" + "="*80)
print("BEST HYPERPARAMETERS FOUND:")
print(f"  Learning rate: {best_config['lr']}")
print(f"  Momentum: {best_config['momentum']}")
print(f"  Weight decay: {best_config['weight_decay']}")
print(f"  Final loss: {best_config['final_loss']:.4f}")
print("="*80)

# ============================================================================
# Q2b: Plot grid search results
# ============================================================================
print("\nPlotting grid search results...")

plt.figure(figsize=(20, 12))
for config_key, losses in grid_search_losses.items():
    plt.plot([1, 2], losses, marker='o', label=config_key, alpha=0.7)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Hyperparameter Grid Search - Training Loss Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.grid(True)
plt.tight_layout()
plt.savefig('grid_search_results.png', dpi=150)
print("✓ Saved grid search plot to grid_search_results.png")

sorted_results = sorted(grid_search_results, key=lambda x: x['final_loss'])
print("\nTop 5 configurations by final loss:")
for i, config in enumerate(sorted_results[:5]):
    print(f"{i+1}. lr={config['lr']}, momentum={config['momentum']}, "
          f"weight_decay={config['weight_decay']}, final_loss={config['final_loss']:.4f}")

# ============================================================================
# Q3a: Train final CNN model with best hyperparameters
# ============================================================================
print("\n" + "="*80)
print("Q3a: Training final CNN model for 50 epochs...")
print("="*80)

final_cnn_model = CNN()

best_lr = best_config['lr']
best_momentum = best_config['momentum']
best_weight_decay = best_config['weight_decay']

optimizer = optim.SGD(final_cnn_model.parameters(), lr=best_lr, momentum=best_momentum, weight_decay=best_weight_decay)

print(f"Using: lr={best_lr}, momentum={best_momentum}, weight_decay={best_weight_decay}")

training_avg_loss = []
val_avg_loss = []
best_val_acc = 0.0
train_acc = []
val_acc = []

epochs = 50

for epoch in range(epochs):
    # Training
    final_cnn_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for batch in pbar:
        optimizer.zero_grad()
        data, target_labels = batch

        prediction = final_cnn_model(data)
        loss = loss_fn(prediction, target_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = running_loss / len(train_loader)
    training_avg_loss.append(avg_train_loss)
    train_accuracy = 100 * correct / total
    train_acc.append(train_accuracy)

    # Validation
    final_cnn_model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            data, target_labels = batch
            prediction = final_cnn_model(data)
            loss = loss_fn(prediction, target_labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(prediction.data, 1)
            val_total += target_labels.size(0)
            val_correct += (predicted == target_labels).sum().item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_avg_loss.append(avg_val_loss)
    val_accuracy = 100 * val_correct / val_total
    val_acc.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(final_cnn_model.state_dict(), 'best_cnn_model.pth')
        print(f"  --> Model saved! New best validation accuracy: {best_val_acc:.2f}%")

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

# ============================================================================
# Q3b: Plot training curves
# ============================================================================
print("\nPlotting training and validation curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(range(1, epochs+1), training_avg_loss, label='Training Loss', marker='o', markersize=3)
ax1.plot(range(1, epochs+1), val_avg_loss, label='Validation Loss', marker='s', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, epochs+1), train_acc, label='Training Accuracy', marker='o', markersize=3)
ax2.plot(range(1, epochs+1), val_acc, label='Validation Accuracy', marker='s', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=150)
print("✓ Saved training curves to cnn_training_curves.png")

# ============================================================================
# Q3c: Load best model
# ============================================================================
print("\n" + "="*80)
print("Q3c: Loading best CNN model...")
print("="*80)

loaded_model = CNN()
loaded_model.load_state_dict(torch.load('best_cnn_model.pth'))
loaded_model.eval()
print("✓ Best model loaded successfully")

# ============================================================================
# Q3d: Evaluate on test set
# ============================================================================
print("\n" + "="*80)
print("Q3d: Evaluating CNN on test set...")
print("="*80)

test_correct = 0
test_total = 0
test_running_loss = 0.0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        data, target_labels = batch
        prediction = loaded_model(data)
        loss = loss_fn(prediction, target_labels)
        test_running_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        test_total += target_labels.size(0)
        test_correct += (predicted == target_labels).sum().item()

test_accuracy = 100 * test_correct / test_total
test_loss = test_running_loss / len(test_loader)

print("="*80)
print("TEST SET RESULTS (Custom CNN Model)")
print("="*80)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {test_correct}/{test_total}")
print("="*80)

# ============================================================================
# PART 2 - TRANSFER LEARNING WITH RESNET18
# ============================================================================
print("\n" + "="*80)
print("PART 2: TRANSFER LEARNING WITH RESNET18")
print("="*80)

from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T

# ============================================================================
# Q1a: Load pretrained ResNet18 and prepare datasets
# ============================================================================
print("\nQ1a: Loading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT
resnet_model = resnet18(weights=weights)

print(f"✓ Loaded pretrained ResNet18")

# Define ResNet transforms
resnet_train_transforms = T.Compose([
    T.Resize((64, 64)),
    T.RandomHorizontalFlip(p=0.25),
    T.RandomVerticalFlip(p=0.25),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet_test_transforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResNetTrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_id in range(6):
            class_path = os.path.join(root_dir, str(class_id))
            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

resnet_full_train_dataset = ResNetTrafficSignDataset(
    root_dir=train_data_path,
    transform=resnet_train_transforms
)

resnet_test_dataset = ResNetTrafficSignDataset(
    root_dir=test_data_path,
    transform=resnet_test_transforms
)

resnet_train_size = int(0.8 * len(resnet_full_train_dataset))
resnet_val_size = len(resnet_full_train_dataset) - resnet_train_size

resnet_train_dataset, resnet_val_dataset = random_split(
    resnet_full_train_dataset,
    [resnet_train_size, resnet_val_size],
    generator=torch.Generator().manual_seed(42)
)

resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=16, shuffle=True, num_workers=2)
resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=16, shuffle=False, num_workers=2)
resnet_test_loader = DataLoader(resnet_test_dataset, batch_size=16, shuffle=False, num_workers=2)

print(f"ResNet datasets created:")
print(f"  Training: {len(resnet_train_dataset)}")
print(f"  Validation: {len(resnet_val_dataset)}")
print(f"  Test: {len(resnet_test_dataset)}")

# ============================================================================
# Q1b: Freeze parameters and replace final layer
# ============================================================================
print("\nQ1b: Freezing ResNet parameters and replacing final layer...")

for param in resnet_model.parameters():
    param.requires_grad = False

num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 6)

print("Trainable parameters:")
for name, param in resnet_model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

print(f"✓ ResNet18 modified: Final layer has {num_features} -> 6 outputs")

# ============================================================================
# Q1c: Train ResNet18
# ============================================================================
print("\n" + "="*80)
print("Q1c: Training ResNet18 with transfer learning...")
print("="*80)

resnet_loss_fn = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet_model.fc.parameters(), lr=0.001)
resnet_epochs = 20

resnet_training_losses = []
resnet_val_losses = []
resnet_train_accs = []
resnet_val_accs = []
best_resnet_val_acc = 0.0

for epoch in range(resnet_epochs):
    # Training
    resnet_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(resnet_train_loader, desc=f"Epoch {epoch+1}/{resnet_epochs} [Train]")
    for batch in pbar:
        resnet_optimizer.zero_grad()

        data, target_labels = batch
        prediction = resnet_model(data)
        loss = resnet_loss_fn(prediction, target_labels)

        loss.backward()
        resnet_optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(prediction.data, 1)
        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = running_loss / len(resnet_train_loader)
    train_accuracy = 100 * correct / total
    resnet_training_losses.append(avg_train_loss)
    resnet_train_accs.append(train_accuracy)

    # Validation
    resnet_model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in resnet_val_loader:
            data, target_labels = batch
            prediction = resnet_model(data)
            loss = resnet_loss_fn(prediction, target_labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(prediction.data, 1)
            val_total += target_labels.size(0)
            val_correct += (predicted == target_labels).sum().item()

    avg_val_loss = val_running_loss / len(resnet_val_loader)
    val_accuracy = 100 * val_correct / val_total
    resnet_val_losses.append(avg_val_loss)
    resnet_val_accs.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{resnet_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_resnet_val_acc:
        best_resnet_val_acc = val_accuracy
        torch.save(resnet_model.state_dict(), 'best_resnet_model.pth')
        print(f"  --> ResNet model saved! New best validation accuracy: {best_resnet_val_acc:.2f}%")

print("="*80)
print(f"ResNet Training complete! Best validation accuracy: {best_resnet_val_acc:.2f}%")

# Plot ResNet training results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(range(1, resnet_epochs+1), resnet_training_losses, label='Training Loss', marker='o', markersize=3)
ax1.plot(range(1, resnet_epochs+1), resnet_val_losses, label='Validation Loss', marker='s', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('ResNet18 - Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, resnet_epochs+1), resnet_train_accs, label='Training Accuracy', marker='o', markersize=3)
ax2.plot(range(1, resnet_epochs+1), resnet_val_accs, label='Validation Accuracy', marker='s', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('ResNet18 - Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('resnet_training_curves.png', dpi=150)
print("✓ Saved ResNet training curves to resnet_training_curves.png")

# ============================================================================
# Q1d: Evaluate ResNet18 on test set
# ============================================================================
print("\n" + "="*80)
print("Q1d: Evaluating ResNet18 on test set...")
print("="*80)

best_resnet_model = resnet18(weights=None)
num_features = best_resnet_model.fc.in_features
best_resnet_model.fc = nn.Linear(num_features, 6)
best_resnet_model.load_state_dict(torch.load('best_resnet_model.pth'))
best_resnet_model.eval()

test_correct = 0
test_total = 0
test_running_loss = 0.0

with torch.no_grad():
    for batch in tqdm(resnet_test_loader, desc="Testing ResNet"):
        data, target_labels = batch
        prediction = best_resnet_model(data)
        loss = resnet_loss_fn(prediction, target_labels)
        test_running_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        test_total += target_labels.size(0)
        test_correct += (predicted == target_labels).sum().item()

test_accuracy = 100 * test_correct / test_total
test_loss = test_running_loss / len(resnet_test_loader)

print("="*80)
print("TEST SET RESULTS (ResNet18 Transfer Learning)")
print("="*80)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {test_correct}/{test_total}")
print("="*80)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. training_samples.png - Sample training images")
print("  2. grid_search_results.png - Hyperparameter search results")
print("  3. cnn_training_curves.png - Custom CNN training curves")
print("  4. best_cnn_model.pth - Best custom CNN model")
print("  5. resnet_training_curves.png - ResNet18 training curves")
print("  6. best_resnet_model.pth - Best ResNet18 model")
print("="*80)
