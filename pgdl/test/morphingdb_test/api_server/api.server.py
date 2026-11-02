from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
from typing import Dict
import uvicorn

# Initialize the FastAPI application
app = FastAPI(
    title="PyTorch model inference service",
    description="API service that provides PyTorch model loading and image inference"
)

# Global store for loaded models
loaded_models: Dict[str, torch.nn.Module] = {}

# Request body models
class LoadModelRequest(BaseModel):
    model_name: str
    model_path: str

class PredictRequest(BaseModel):
    model_name: str
    image_url: str

# Image preprocessing transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load model endpoint
@app.post("/load_model/")
async def load_model(request: LoadModelRequest):
    try:
        # Check whether the model file exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at {request.model_path}")
        
        # Load the model
        model = torch.load(request.model_path, weights_only=False)
        model = model.to("cuda:0")
        model.eval()  # set to eval mode
        
        # Store in the global dict
        loaded_models[request.model_name] = model
        
        return {
            "status": "success",
            "message": f"Model '{request.model_name}' loaded successfully from {request.model_path}",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Inference endpoint
@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        # Check whether the model is loaded
        if request.model_name not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not loaded. Please load it first using /load_model/"
            )
        
        # Get the model
        model = loaded_models[request.model_name]
        
        # Load and preprocess the image
        # If it is a local path, open the file directly
        if request.image_url.startswith("/"):
            image = Image.open(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid image URL. Please provide a valid local path.")
        
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to("cuda:0")  # Move the input tensor to the GPU
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "image_url": request.image_url,
            "predicted_class": predicted_class.item(),
            "confidence": torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# List the loaded models
@app.get("/loaded_models/")
async def get_loaded_models():
    return {
        "loaded_models": list(loaded_models.keys()),
        "count": len(loaded_models)
    }



@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)