"""
Utility script to export PyTorch models to ONNX format.
"""
import torch
import os
from model import load_model

def export_to_onnx(pytorch_model_path="model.pth", onnx_model_path="model.onnx"):
    print(f"Loading PyTorch model from {pytorch_model_path}...")
    try:
        model = load_model(pytorch_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Create dummy input with the correct shape (Batch Size, Channels, Height, Width)
    # EfficientNetV2-S uses 224x224 input
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to {onnx_model_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=14,          # Opset version supported by most runtimes
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    pytorch_path = os.path.join(current_dir, "model.pth")
    onnx_path = os.path.join(current_dir, "model.onnx")
    export_to_onnx(pytorch_path, onnx_path)
