import torch
from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# --- STEP 1: THE SINGLETON LOADER ---
class DeepfakeModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeModel, cls).__new__(cls)
            # Defining architecture (ResNet18)
            cls._model = models.resnet18()
            num_ftrs = cls._model.fc.in_features
            cls._model.fc = torch.nn.Linear(num_ftrs, 2)
            
            # Load your weights
            cls._model.load_state_dict(torch.load("trained_deepfake_detector.pth", map_location=torch.device('cpu')))
            cls._model.eval()
            print("🚀 Model loaded into memory successfully (Singleton).")
        return cls._instance

    @property
    def model(self):
        return self._model

# Initialize the singleton instance
model_loader = DeepfakeModel()

# --- STEP 2: THE HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """
    Standard health check for production monitoring.
    """
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": "cpu", #  will ssChange to 'cuda' if using GPU
        "architecture": "ResNet18"
    }

# --- STEP 3:  PREDICTION ROUTE ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image from the Singleton
    model = model_loader.model
    
    # Preprocessing
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    #  transformation logic here
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    label = "Fake" if prediction == 1 else "Real"
    return {"result": label}