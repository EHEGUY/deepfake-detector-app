import os
import io
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Forensic API", version="2.0.0")

# 1. THE BRIDGE (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CUSTOM ARCHITECTURE REBUILDER
def build_4channel_resnet() -> nn.Module:
    """Rebuilds the custom 4-channel architecture so the new weights fit."""
    model = models.resnet18(weights=None)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=4, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding, 
        bias=False
    )
    model.conv1 = new_conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# 3. SINGLETON LOADER 
class DeepfakeModel:
    _instance = None
    _model = None
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeModel, cls).__new__(cls)
            cls._initialize_model()
        return cls._instance

    @classmethod
    def _initialize_model(cls):
        logger.info(f"Initializing 4-Channel ResNet18 on {cls._device}...")
        cls._model = build_4channel_resnet()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "forensic_resnet18_4channel.pth")
        
        if not os.path.exists(model_path):
            logger.warning(f"Custom Weights not found at {model_path}. Waiting for training to finish...")
            return

        cls._model.load_state_dict(torch.load(model_path, map_location=cls._device, weights_only=True))
        cls._model.to(cls._device)
        cls._model.eval()
        logger.info("Forensic Model loaded successfully.")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

inference_engine = DeepfakeModel()

# Laplacian kernel for edge/noise detection
laplacian_kernel = torch.tensor([
    [[[0.0,  1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0,  1.0, 0.0]]]
], dtype=torch.float32).to(inference_engine.device)

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "device": str(inference_engine.device),
        "architecture": "Custom 4-Channel ResNet18"
    }

# 4. PREDICTION ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 1. Base Preprocessing (Resize and convert to [0, 1] tensor)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        rgb_tensor = preprocess(image).to(inference_engine.device)
        
        # 2. Generate the 4th Channel (Edge Map)
        grayscale = 0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]
        edge_map = F.conv2d(grayscale.unsqueeze(0), laplacian_kernel, padding=1).squeeze(0)
        
        # 3.  THE FIX: Normalize RGB to match the trainer 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_normalized = normalize(rgb_tensor)
        
        # 4. Stack RGB + Edge Map and add batch dimension [1, 4, 224, 224]
        four_channel_tensor = torch.cat((rgb_normalized, edge_map), dim=0).unsqueeze(0)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if inference_engine.device.type == 'cuda' else 'cpu'):
                output = inference_engine.model(four_channel_tensor)
                probabilities = F.softmax(output, dim=1)

                # Log "Raw Thoughts" to your console for debugging
                print(f"RAW THOUGHTS: Real={probabilities[0][0]:.4f} | Fake={probabilities[0][1]:.4f}")

                confidence, prediction = torch.max(probabilities, dim=1)
        
        return {
            "result": "Fake" if prediction.item() == 1 else "Real",
            "confidence": round(confidence.item() * 100, 2)
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")