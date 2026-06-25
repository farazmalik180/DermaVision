import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Import custom modules
from model import load_model, CLASSES, generate_gradcam
from utils import check_image_blur, preprocess_image

app = FastAPI(
    title="DermaVision API",
    description="Backend API for skin lesion classification and Grad-CAM visualization.",
    version="1.0.0"
)

# Configure CORS so the React frontend (running on localhost) can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the PyTorch model globally at startup
# Looks for backend/model.pth in the same directory, or uses baseline initialized weights
weights_file = os.path.join(os.path.dirname(__file__), "model.pth")
model = load_model(weights_file)

@app.get("/health")
def health_check():
    """Simple API status health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a skin lesion image, performs blur detection, processes predictions,
    and returns class, confidence, risk level, and Grad-CAM heatmap overlay.
    """
    # 1. Validate file content type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG image."
        )

    try:
        # Read the file content
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    # 2. Perform Blur Detection
    is_blurry, variance, err = check_image_blur(contents, threshold=50.0)
    if err:
        raise HTTPException(status_code=500, detail=err)
    if is_blurry:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too blurry (Laplacian variance: {variance:.1f}). Please upload a clear photo with good lighting and in-focus detail."
        )

    try:
        # 3. Preprocess the image for the model
        input_tensor, original_img_np = preprocess_image(contents)
        
        # 4. Perform Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)
            
        # Get highest probability class
        confidence, class_idx = torch.max(probabilities, dim=0)
        confidence = float(confidence.item())
        class_idx = int(class_idx.item())
        
        # Map to class details
        predicted_class = CLASSES[class_idx]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running model inference: {str(e)}"
        )

    # 5. Generate Grad-CAM image (requires gradients, so we enable it momentarily)
    # Grad-CAM needs to run forward/backward pass, so we enable grad for this operation
    try:
        with torch.enable_grad():
            gradcam_base64 = generate_gradcam(model, input_tensor, original_img_np, class_idx)
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        gradcam_base64 = ""

    # 6. Return response
    return {
        "label": predicted_class["name"],
        "confidence": confidence,
        "risk_level": predicted_class["risk_level"],
        "description": predicted_class["desc"],
        "gradcam_image_base64": f"data:image/jpeg;base64,{gradcam_base64}" if gradcam_base64 else None
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
