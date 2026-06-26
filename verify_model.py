import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from model import load_model

weights_path = os.path.join(os.path.dirname(__file__), "backend", "model.pth")
print(f"Loading weights from {weights_path}")
model = load_model(weights_path)
print("Model loaded successfully!")
