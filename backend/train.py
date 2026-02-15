import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Define paths
train_dir = './dataset/train'
val_dir = './dataset/validation'
test_dir = './dataset/test'

# ==========================================
# 3. DEFINE TRANSFORMS (The Augmentation Upgrade)
# ==========================================

# A. Training Transform (Hard Practice)
# Randomly flips, rotates, and changes lighting so the model doesn't memorizes.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),               # 50% chance to mirror left/right
    transforms.RandomRotation(degrees=10),                # Tilt up to 10 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Slight lighting changes
    transforms.ToTensor()
])

# B. Validation & Test Transform (The Real Match)
# Clean and standard. No tricks here, just what the model will see in the real world.
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. Load datasets (Applying the specific transforms)
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 6. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 7. Training loop with Progress Bar
epochs = 5

print("Starting training with Data Augmentation...")
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
    
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
        
    avg_train_loss = running_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False)
    
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{epochs}] Completed | Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

print("Training finished!")

# 8. Save the model
torch.save(model.state_dict(), 'trained_deepfake_detector.pth')
print("--- MODEL SAVED SUCCESSFULLY ---")