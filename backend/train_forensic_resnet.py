import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ForensicDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)

        for label, category in enumerate(["real", "fake"]):
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir): continue
            for img_name in os.listdir(category_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(category_dir, img_name))
                    self.labels.append(label)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Use the transform (which now includes heavy augmentation)
        rgb_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)

        # 1. Edge Map Calculation
        grayscale = 0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]
        edge_map = F.conv2d(grayscale.unsqueeze(0), self.laplacian_kernel, padding=1).squeeze(0)
        
        # 2. RGB Normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_normalized = normalize(rgb_tensor)

        return torch.cat((rgb_normalized, edge_map), dim=0), torch.tensor(label, dtype=torch.long)

def build_4channel_resnet() -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(4, old_conv.out_channels, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = old_conv.weight
        nn.init.kaiming_normal_(model.conv1.weight[:, 3:4, :, :])
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def train_forensic_model(data_dir: str, epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  HEAVY AUGMENTATION: Stop the model from memorizing pixel locations
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    dataset = ForensicDataset(root_dir=data_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = build_4channel_resnet().to(device)
    
    #  WEIGHTED LOSS: Force the model to prioritize finding Fakes (Index 1)
    # Missing a fake is now 5x more "painful" for the model than missing a real image.
    weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 20 == 0:
                logger.info(f"Epoch {epoch+1} | Acc: {100.*correct/total:.2f}% | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "forensic_resnet18_4channel.pth")
    logger.info("Training Finished with Weighted Loss.")

if __name__ == "__main__":
    train_forensic_model(r"C:\Users\siddt\OneDrive\Desktop\deepfake-detector-app\backend\dataset\train", epochs=1)