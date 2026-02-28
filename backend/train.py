import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pathlib import Path

# 1. Setup paths
base_dir = Path(r"C:\Users\siddt\OneDrive\Desktop\deepfake\dataset")
model_path = Path(__file__).resolve().parent / "forensic_resnet18_4channel.pth"
save_path = Path(__file__).resolve().parent / "forensic_resnet18_fast_tuned.pth"

# 2. Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Updated Helper to handle the "validation" folder name
def load_ds(set_num, split_name):
    # This matches: Data Set X / Data Set X / validation (or train)
    p = base_dir / f"Data Set {set_num}" / f"Data Set {set_num}" / split_name
    if not p.exists():
        raise FileNotFoundError(f"❌ Path not found: {p}")
    print(f" Loaded: {p}")
    return datasets.ImageFolder(p, transform=transform)

print("--- Starting Dataset Check ---")
try:
    # Load Training Sets (2, 3, 4)
    train_sets = [load_ds(2, "train"), load_ds(3, "train"), load_ds(4, "train")]
    full_train_ds = ConcatDataset(train_sets)

    # Load Validation Sets 
    val_sets = [load_ds(2, "validation"), load_ds(3, "validation"), load_ds(4, "validation")]
    full_val_ds = ConcatDataset(val_sets)
    
    print(f"Total training images: {len(full_train_ds)}")
    print(f"Total validation images: {len(full_val_ds)}")

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nDouble check your folder names in File Explorer!")
    exit()

# : Only train on 1500 images to keep it under 20 mins ---
subset_indices = torch.randperm(len(full_train_ds))[:1500] 
train_subset = Subset(full_train_ds, subset_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(full_val_ds, batch_size=32, shuffle=False)

# 3. Model Setup (4-channel support)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.models import resnet18
model = resnet18(weights=None)

# ADJUST FOR 4-CHANNELS
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 2) 

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("\nSuccessfully loaded 4-channel weights.")

model = model.to(device)

# 4. Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 5. Training Loop (2 Epochs)
print("\nStarting Training (Target: < 20 mins)...")
for epoch in range(2):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Add 4th channel to 3-channel images
        if inputs.shape[1] == 3:
            dummy = torch.zeros((inputs.shape[0], 1, 224, 224)).to(device)
            inputs = torch.cat((inputs.to(device), dummy), dim=1)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

# 6. Save
torch.save(model.state_dict(), save_path)
print(f"\n Done! New model saved as: {save_path}")