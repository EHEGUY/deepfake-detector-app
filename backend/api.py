import torch
import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image

# Initialize 
app = FastAPI()

# 1. THE BRIDGE (CORS) - Fixed: No duplicates
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.  SINGLETON LOADER 
class DeepfakeModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeModel, cls).__new__(cls)
            # Architecture
            cls._model = models.resnet18()
            num_ftrs = cls._model.fc.in_features
            cls._model.fc = torch.nn.Linear(num_ftrs, 2)
            
            # Load Weights
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODEL_PATH = os.path.join(BASE_DIR, "trained_deepfake_detector.pth")
            
            cls._model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
            cls._model.eval()
            print("🚀 Model loaded into memory successfully (Singleton).")
        return cls._instance

    @property
    def model(self):
        return self._model

model_loader = DeepfakeModel()

#  3 HEALTH CHECK
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": "cpu",
        "architecture": "ResNet18"
    }

# 4. PREDICTION ROUTE well  Now calculates real Confidence Score
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = model_loader.model
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        # STEP 4.1: Well This Should Calculate Probability using Softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    # STEP 4.2: Convert to percentage (0.98 -> 98.0)
    score = confidence.item() * 100
    label = "Fake" if prediction.item() == 1 else "Real"
    
    return {
        "result": label,
        "confidence": score  # Return confidence as percentage
        
    }