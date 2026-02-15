from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI()

# This is CRITICAL. It allows a v0 website to talk to this local Python server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the Model into memory when the server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading AI Brain onto: {device}...")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('trained_deepfake_detector.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ['Fake', 'Real']

# 2. Define the Endpoint (The "Waiter")
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Process it through the AI
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        
    result = classes[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    
    # Send the data back as a clean JSON package
    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }