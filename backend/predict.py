import torch
from torchvision import transforms, models
from PIL import Image
# (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Loadas the ResNet18 architecture
# We set weights=None because we don't want the generic PyTorch brain, we want your custom-trained brain

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# 3. Load your custom-trained weights
model.load_state_dict(torch.load('trained_deepfake_detector.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Set to evaluation mode (turns off learning)

# PyTorch ImageFolder sorts folders alphabetically: 'fake' is 0, 'real' is 1
classes = ['Fake', 'Real']

# 4. Image transformations (MUST match exactly what you used in train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


image_path = r"C:\Users\siddt\OneDrive\Desktop\deepfake\dataset\test\fake\fake_5.jpg"
try:
    # Open the image and convert it to the format the model expects
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 6. Makse the Predictio 
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        
    result = classes[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    
    print("\n" + "="*40)
    print(f"Image: {image_path}")
    print(f"Prediction: {result.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

except FileNotFoundError:
    print(f"\n[ERROR] I couldn't find an image at: {image_path}")
    print("Please check your 'dataset/test' folder, copy the exact name of a picture, and update the 'image_path' variable in the code!")