"""
FastAPI backend for DermaVision application.
"""
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import onnxruntime as ort
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from huggingface_hub import hf_hub_download

# Load the PyTorch model globally at startup
weights_file = os.path.join(os.path.dirname(__file__), "model.pth")
onnx_weights_file = os.path.join(os.path.dirname(__file__), "model.onnx")

if not os.path.exists(weights_file):
    print("Downloading model.pth from Hugging Face...")
    try:
        hf_hub_download(repo_id="MF180/DermaVision-EfficientNet", filename="model.pth", local_dir=os.path.dirname(__file__))
    except Exception as e:
        print(f"Error downloading model: {e}")

model = load_model(weights_file) # Always load PyTorch model for Grad-CAM

if os.path.exists(onnx_weights_file):
    print("Loading ONNX model for faster inference...")
    ort_session = ort.InferenceSession(onnx_weights_file)
else:
    ort_session = None

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
        if ort_session is not None:
            # Use ONNX Runtime for faster inference
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = torch.tensor(ort_outs[0])
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        else:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0)
            
        # Get highest probability class
        confidence, class_idx = torch.max(probabilities, dim=0)
        confidence = float(confidence.item())
        class_idx = int(class_idx.item())
        
        # --- FIX: Apply 0.2 classification threshold for Melanoma (Index 0) ---
        melanoma_prob = float(probabilities[0].item())
        if melanoma_prob > 0.2:
            class_idx = 0
            confidence = melanoma_prob
        
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

# --- Optional Static File Serving for Hugging Face Spaces ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    print(f"Static directory found at {static_dir}. Serving React frontend...")
    # Serve index.html at the root
    @app.get("/")
    async def serve_react_app():
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    # Mount all other static files (assets, css, js)
    app.mount("/", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
