import io
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# 1. SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Forensic API", version="3.0.0")

# 2. CORS (Connects  Frontend to this Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. THE SINGLETON ENGINE (Loads Model Once)
class DeepfakeModel:
    _instance = None
    _model = None
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _kernel = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeModel, cls).__new__(cls)
            cls._initialize_engine()
        return cls._instance

    @classmethod
    def _initialize_engine(cls):
        """Builds and loads the model into RAM exactly once at startup."""
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "forensic_resnet18_fast_tuned.pth"
        
        logger.info(f" Initializing Singleton ResNet18 on {cls._device}")
        
        # Build 4-channel Architecture
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Load weights
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=cls._device))
            logger.info(f" Weights loaded: {model_path.name}")
        else:
            logger.error(f" CRITICAL: Weights not found at {model_path}")

        cls._model = model.to(cls._device)
        cls._model.eval()

        # Initialize Laplacian Kernel for noise/artifact detection
        cls._kernel = torch.tensor([
            [[[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]]
        ], dtype=torch.float32).to(cls._device)

    @property
    def model(self): return self._model
    
    @property
    def device(self): return self._device
    
    @property
    def kernel(self): return self._kernel

# Instantiate the singleton immediately
engine = DeepfakeModel()

# 4. PREDICTION LOGIC
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Load Image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocessing (Resize and Tensor)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        rgb_tensor = transform(image).to(engine.device)
        
        # Generate 4th Channel (Edge Map)
        grayscale = 0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]
        edge_map = F.conv2d(grayscale.unsqueeze(0), engine.kernel, padding=1).squeeze(0)
        
        # Normalize RGB to match the trainer
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_norm = normalize(rgb_tensor)
        
        # Stack to 4 channels: [RGB + Edge] -> [1, 4, 224, 224]
        final_input = torch.cat((rgb_norm, edge_map), dim=0).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = engine.model(final_input)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        # Result mapping
        classes = ["Fake", "Real"]
        result = classes[pred.item()]
        
        logger.info(f"Prediction: {result} ({conf.item()*100:.2f}%)")
        
        return {
            "result": result,
            "confidence": round(conf.item() * 100, 2),
            "raw_scores": {
                "fake": round(probs[0][0].item(), 4),
                "real": round(probs[0][1].item(), 4)
            }
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupt image file.")
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)