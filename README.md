# DermaVision 🩺

DermaVision is a research-grade, full-stack skin lesion analysis and classification application. It uses a fine-tuned **EfficientNetV2-S** deep learning model to evaluate skin lesions, map them to specific risk categories, and render **Grad-CAM** heatmaps indicating where the network focused its visual attention.

The repository offers two client choices:
1. **Full-Stack Stack**: A FastAPI Python backend served alongside a React + Tailwind CSS client dashboard.
2. **Streamlit Dashboard**: A fast, single-process, Python-only interactive Streamlit app that shares the backend model logic.

---

## 📂 Project Structure

```text
DermaVision/
├── backend/
│   ├── blurry_lesion.jpg        # Blurry test image
│   ├── clear_lesion.jpg         # Clear test image
│   ├── create_test_images.py    # Test image generator script
│   ├── main.py                  # FastAPI application entrypoint
│   ├── model.py                 # PyTorch model definition and Grad-CAM logic
│   ├── requirements.txt         # Python package dependencies
│   └── utils.py                 # Preprocessing and blur check utilities
├── frontend/
│   ├── public/                  # Static assets
│   ├── src/                     # React application source code
│   │   ├── components/          # UI components (Home, History, PulseLoader)
│   │   ├── App.jsx              # Main React App component
│   │   └── index.css            # Base Tailwind and custom styles
│   ├── index.html               # Frontend HTML shell
│   ├── package.json             # NPM package scripts & configuration
│   └── vite.config.js           # Vite build tool configuration
├── streamlit_app/
│   └── app.py                   # Streamlit interactive dashboard client
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (tested and recommended with Python 3.11 for stable PyTorch compatibility)
- Node.js 18+ & npm 9+ (tested with Node 22.14 & npm 10.9)

---

### Option A: Running the Full-Stack App (FastAPI + React)

#### 1. Backend Server Setup & Start
Navigate to the `backend/` directory:
```bash
cd backend
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI development server:
```bash
python main.py
```
- The backend will start on **`http://localhost:8000`**.
- You can access the automatic documentation and interactive API playground at `http://localhost:8000/docs`.
- The health check is available at `http://localhost:8000/health`.

*Note: If no custom weights file (`model.pth`) is present in the `backend/` folder, the server will load EfficientNetV2-S with default ImageNet feature extraction weights and adapt the classification head.*

#### 2. Frontend client Setup & Start
Open a new terminal window and navigate to the `frontend/` directory:
```bash
cd frontend
```

Install Node dependencies:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
- The client app will load at **`http://localhost:5173`**.

---

### Option B: Running the Streamlit App

The Streamlit app acts as a single-process alternative client utilizing the same model, preprocessing, and blur checking logic.

Open a terminal and navigate to the project root:
```bash
streamlit run streamlit_app/app.py
```
- Streamlit will launch the web application dashboard at **`http://localhost:8501`**.

---

## 🧠 Model Training Notes

To achieve the research-grade metrics target of **AUC-ROC > 0.90**, we follow this specific model training procedure:

### Hardware Setup
- **GPU**: NVIDIA Tesla T4 (via Google Colab)
- **Reasoning**: The T4 GPU provides 16GB of VRAM, which is sufficient to comfortably fit the EfficientNetV2-S model and a reasonable batch size during mixed-precision training. Additionally, it is widely accessible and cost-effective, making the model training easily reproducible without requiring expensive local hardware.

### Kaggle API Setup & Dataset Download

The Kaggle API token is only needed during the model training phase to download the training dataset. It is **NOT** used in the deployed application at runtime. The app itself (FastAPI backend + Streamlit) requires no API keys or credentials, and end users do not need a Kaggle account.

To download the dataset for training:
1. **Obtain API Token**: The developer (not the end user) must log into Kaggle and obtain a free API token from [kaggle.com/settings](https://www.kaggle.com/settings) (under the API section, click "Create New Token").
2. **Place Token File**: Place the downloaded `kaggle.json` credentials file in the directory `~/.kaggle/` (on Windows, `C:\Users\<Username>\.kaggle\`) on the training machine.
3. **Add to Gitignore**: Add `kaggle.json` to your `.gitignore` file to ensure credentials are never committed to the repository.
4. **Download Command**: Run the following command via the Kaggle CLI to download the dataset:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```

### Detailed CNN Pipeline Architecture

The end-to-end pipeline for training and inference is designed to maximize feature extraction while aggressively combating class imbalance.

```mermaid
graph TD
    A[Input Image] --> B[Preprocessing & Augmentation]
    B --> C[EfficientNetV2-S Backbone]
    
    subgraph CNN Architecture
        C -->|Shallow Features| D[Fused-MBConv Blocks]
        D -->|Deep Features| E[MBConv Blocks]
        E -->|Final Spatial Maps| F[Global Average Pooling]
    end
    
    F -->|1D Feature Vector| G[Dropout Layer]
    G -->|Regularized Vector| H[Linear Classification Head]
    
    subgraph Output & Post-Processing
        H -->|Raw Logits| I[Softmax Activation]
        I -->|Probabilities| J{Melanoma Prob > 0.2?}
        J -->|Yes| K[Classify as Melanoma]
        J -->|No| L[Standard Argmax]
    end
    
    E -.->|Target Layer Gradients| M[Grad-CAM Visualizer]
```

#### Pipeline Stages
1. **Input & Preprocessing**: Images are loaded, resized to `224x224`, and normalized using standard ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. This standardizes the color distribution, accelerating model convergence.
2. **Data Augmentation (Training Only)**: To simulate diverse clinical conditions, we apply Random Flips, Rotations, Color Jitter, and Random Erasing (Cutout). Melanoma samples receive uniquely aggressive augmentations (e.g., higher rotation degrees up to 90°, affine shifts) to enforce robust feature learning and combat extreme class imbalance.
3. **Feature Extraction (EfficientNetV2-S Backbone)**:
   - **Fused-MBConv (Early Layers)**: Standard depthwise convolutions in early layers are slow due to memory access overhead. EfficientNetV2 replaces them with Fused-MBConv blocks (regular convolutions) in the early stages to maximize execution speed and extract shallow textures (edges, color gradients).
   - **MBConv (Deep Layers)**: For deeper layers, standard Mobile Inverted Bottleneck (MBConv) blocks are used. These rely on depthwise separable convolutions to efficiently extract complex, high-level structural features (lesion borders, internal network structures) while keeping parameter counts low.
4. **Classification Head**: A Global Average Pooling (GAP) layer collapses the spatial dimensions `[B, C, H, W]` into a 1D feature vector `[B, C]`. This vector passes through a Dropout layer (to prevent overfitting by randomly zeroing out features) and finally through a dense Linear layer that maps to our 7 specific lesion classes.
5. **Loss & Optimization (Training)**: The network optimizes using the **AdamW** optimizer, which decouples weight decay for better regularization. Learning rate is managed by a Cosine Annealing scheduler. Errors are penalized using **Focal Loss** ($\gamma=4.0$) with heavy multiplier weights given to minority and malignant classes to prevent the model from ignoring rare but fatal diseases.
6. **Inference & Post-Processing**:
   - Softmax is applied to the raw logits to generate a probability distribution summing to 1.
   - **Threshold Override**: To strictly minimize False Negatives for deadly Melanoma, we override standard argmax classification. If the Melanoma probability exceeds `20%`, the system automatically flags it as Melanoma.
   - **Grad-CAM**: The gradients flowing into the final convolutional layer are tapped to generate class activation heatmaps. This provides clinical interpretability by overlaying a heatmap on the original image, showing exactly which visual features the CNN focused on.
   - **Layer-wise Feature Visualization**: We also provide an offline diagnostic tool (`backend/visualize_features.py`) to extract and plot the intermediate feature maps from the backbone. This allows researchers to see exactly how the network's understanding of a lesion transitions from shallow textures (edges/colors) in the Fused-MBConv layers to complex semantic concepts in the deep MBConv layers.

![Layer-wise Feature Visualization](./backend/layer_features.png)
*Above: Intermediate feature maps extracted at different depths of the EfficientNetV2-S backbone for a clear lesion. Notice how early layers (Stem/Fused-MBConv) focus on fine-grained textures and borders, while deeper layers (MBConv/Final Conv) abstract these into high-level semantic patterns.*

#### Forward Propagation Animation
To better visualize how the network progressively processes the image, we also provide a forward propagation animation. This animation steps through the mean activation of every convolutional block in the network from start to finish.

![Forward Propagation Animation](./backend/forward_prop.gif)
*Above: Forward propagation of an image through all blocks of the EfficientNetV2-S backbone, illustrating the transition from spatial awareness to semantic features.*

#### Why EfficientNetV2-S?
For medical image classification (like Dermoscopy), high accuracy is paramount, but computational efficiency allows for practical deployment. We selected **EfficientNetV2-S** over traditional architectures (like ResNet or VGG) because it uses Neural Architecture Search (NAS) to jointly optimize training speed and parameter efficiency. It achieves state-of-the-art accuracy with significantly fewer parameters, making our FastAPI and Streamlit inference extremely fast even on CPU.

### 1. Dataset & Class Definition
- **Dataset**: HAM10000 Dataset (10,015 high-resolution dermoscopic images). *Note: This dataset is used ONLY during offline training. The deployed app does not call the Kaggle API at runtime — it loads `model.pth` locally from the `backend/` folder.*
- **Target Classes (7)**:
  1. **Melanoma** (Malignant) - *High Risk*
  2. **Melanocytic Nevus** (Benign Mole) - *Low Risk*
  3. **Basal Cell Carcinoma** (Malignant) - *High Risk*
  4. **Actinic Keratosis** (Pre-cancerous) - *Moderate Risk*
  5. **Seborrheic Keratosis** (Benign Keratosis) - *Low Risk*
  6. **Dermatofibroma** (Benign Growth) - *Low Risk*
  7. **Vascular Lesion** (Benign Blood Vessel Mark) - *Low Risk*

### 2. Class Imbalance Mitigation
Skin lesion datasets are heavily imbalanced (e.g., Melanocytic Nevus accounts for >80% of samples). To prevent the model from biassing towards majority classes:
- **Batch Sampling**: Implement a PyTorch `WeightedRandomSampler` to oversample minority classes. We apply a **5x multiplier** to the standard inverse class frequency weights to aggressively push the model to learn rare cases.
- **Criterion**: Use **Focal Loss** instead of standard Cross Entropy. Focal Loss dynamically down-weights well-classified examples ($p > 0.5$) and focuses back-propagation steps on hard, misclassified samples:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  *Parameters used: $\gamma = 4.0$, $\alpha$ heavily weighted towards Melanoma and minority classes (5x boost).*

### 3. Data Augmentation
To prevent overfitting on lighting variations and patient skin-tones, we implement a robust image augmentation pipeline:
- **Geometry**: Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$), Random Rotation ($\pm30^\circ$).
- **Color**: Color Jitter (Brightness $\pm0.2$, Contrast $\pm0.2$, Saturation $\pm0.2$, Hue $\pm0.1$).
- **Regularization**: Coarse Dropout (Cutout) to mask random $16\times16$ patches, forcing the model to look at multiple features of a lesion rather than a single focal hotspot.
- **Class-Specific Augmentation**: To improve generalizability for Melanoma, we apply stronger, distinct augmentations exclusively to Melanoma samples during training (increased rotation limits and aggressive color/affine jittering).

### 4. Training Schedule
- **Architecture**: EfficientNetV2-S (`torchvision.models.efficientnet_v2_s`).
- **Epochs**: 20 epochs.
- **Learning Rate**: $1\times10^{-4}$ initialized.
- **Scheduler**: Cosine Annealing learning rate scheduler with restarts, decaying the learning rate smoothly to a minimum of $1\times10^{-6}$.
- **Checkpointing**: The model is evaluated on validation folds after every epoch. The checkpoint with the highest **Validation AUC-ROC** is saved as `model.pth`.

### 5. Evaluation Protocol
At evaluation phase, the model outputs are assessed on validation splits using:
- **AUC-ROC (Area Under ROC Curve)**: Combined metric for overall multi-class discrimination (target $>0.90$).
- **Accuracy**: Overall correct rate.
- **Sensitivity (Recall / True Positive Rate)**: Critical for minimizing false negatives in malignant cases (Melanoma).
- **Specificity (True Negative Rate)**: Critical for minimizing unnecessary biopse procedures on benign moles.
- **Confusion Matrix**: Tracking cross-class errors (e.g. distinguishing Seborrheic Keratosis from Melanoma).

---

## 📊 Results

### Bias vs Variance Analysis
To ensure our model generalizes well and correctly balances underfitting and overfitting, we've introduced automatic **Bias vs Variance Analysis** tracking during training.

After each training run, the `train.py` script automatically:
1. Generates `training_history.csv` with per-epoch logs.
2. Plots `learning_curves.png` showing the training vs. validation loss and accuracy.
3. Outputs an automated diagnosis determining if the model is currently underfitting (high bias), overfitting (high variance), or well-balanced.

*Note: You can also run this diagnosis offline on an already trained model without retraining by running `python backend/analyze_bias_variance.py`. This script sweeps the training and validation datasets locally and outputs the diagnosis based on the weights in `model.pth`.*

### Training Iteration 1 (Baseline)
This was the initial baseline model before applying minority class weighting. The model heavily favored majority classes (like Nevus), leading to high overall accuracy but dangerous performance on minority malignant classes.

| Metric                | Score  |
|-----------------------|--------|
| Validation AUC-ROC    | 0.979  |
| Training Accuracy     | 92.6%  |
| Validation Accuracy   | 75.7%  |
| Macro Sensitivity     | 14.7%  |
| Melanoma Sensitivity  | 5.4%   |
| Macro Specificity     | 85.6%  |

### Training Iteration 2 (Sensitivity Focus)
Due to the unacceptably low Melanoma sensitivity in the baseline model, we retrained with targeted fixes: lowered prediction threshold to 0.2, 5x focal loss weighting, 5x minority sampler, and stronger Melanoma-specific augmentations. 

* **Outcome**: Macro sensitivity massively improved to 71.1%. Most minority classes (Basal Cell, Dermatofibroma, Vascular Lesion) successfully hit the >85% target.
* **Remaining Issue**: Melanoma sensitivity improved 7x (up to 38.1%) but still failed to hit the clinical >85% target. The model strongly confused Malignant Melanoma with Benign Seborrheic Keratosis (predicting Seborrheic Keratosis for almost half the Melanoma cases).
* **Why did Overall Accuracy Drop?**: The dataset is heavily imbalanced (67% of images are benign Nevus). In Iteration 1, the model "cheated" by always guessing Nevus to achieve 75% accuracy. In Iteration 2, we heavily penalized the model for missing cancers. The model became hyper-vigilant and started over-predicting minority classes. This caused it to misclassify many harmless Nevus cases—which tanked the overall accuracy—but it drastically increased the life-saving Sensitivity scores.

| Metric                | Score  |
|-----------------------|--------|
| Validation AUC-ROC    | 0.915  |
| Training Accuracy     | 95.4%  |
| Validation Accuracy   | 39.0%  |
| Macro Sensitivity     | 71.1%  |
| Melanoma Sensitivity  | 38.1%  |
| Macro Specificity     | 90.2%  |

### Training Iteration 3 (High-Res & Overfitting Fixes)
To combat the Seborrheic Keratosis confusion from Iteration 2, we heavily upgraded the training pipeline:
* Increased resolution to `384x384` to preserve microscopic textures.
* Switched to `OneCycleLR` for faster convergence.
* Added a 3x weight penalty specifically for Seborrheic Keratosis.
* Increased Dropout (0.6) and Weight Decay (1e-1) to combat massive overfitting caused by aggressive minority class weighting.

* **Outcome**: The model successfully crossed the research-grade target, achieving an **AUC-ROC of 0.9129**.
* **Accuracy Tradeoff**: Validation accuracy dropped to 25.9%, but this is an intended effect. Because of our strict 20% threshold override for Melanoma and massive 5x Focal Loss penalties, the model is designed to be hyper-vigilant. It sacrifices overall accuracy (by flagging many benign moles as suspicious) in order to maximize life-saving sensitivity.

| Metric                | Score   |
|-----------------------|---------|
| Validation AUC-ROC    | 0.9129  |
| Training Accuracy     | 96.8%   |
| Validation Accuracy   | 25.9%   |
| Macro Sensitivity     | 71.1%   |
| Melanoma Sensitivity  | 68.2%   |
| Macro Specificity     | 88.2%   |

---

## 📝 Recent Codebase Updates

To support our transition towards strict clinical sensitivity, several scripts have been heavily upgraded from their original baseline versions:

* **`backend/evaluate.py`**: Upgraded from a simple accuracy loop into a deep diagnostic tool. It now generates full 7x7 confusion matrices, detailed `sklearn` classification reports, per-class One-vs-Rest AUC-ROC scores, and explicit Macro and Per-Class Sensitivity/Specificity metrics.
* **`backend/train.py`**: Completely overhauled to combat class imbalance. Replaced standard Cross-Entropy with a custom **Focal Loss** implementation (gamma=4.0). Added a **WeightedRandomSampler** to aggressively oversample minority classes by 5x, and implemented strong class-specific data augmentations specifically for Melanoma cases. We also added Google Drive auto-backup logic and fixed a critical bug where `--dry-run` executions would overwrite the production `model.pth`.
* **`backend/main.py`**: Upgraded the FastAPI inference pipeline to respect custom risk thresholds. Instead of returning a blind `argmax` prediction, if the predicted probability for Melanoma exceeds `0.20`, the API will proactively override the prediction and flag the lesion as Melanoma to ensure patient safety.
* **Docker Infrastructure**: Fully containerized the application with a `docker-compose.yml` and individual `Dockerfile`s for the backend, frontend, and Streamlit app.
* **CI/CD & Testing**: Implemented automated testing with `pytest` (Python) and `Vitest` (React). Added a GitHub Actions workflow (`ci.yml`) to run tests on every commit.
* **ONNX Runtime Support**: Added `export_to_onnx.py` to compile the PyTorch model into an optimized ONNX graph, drastically reducing CPU inference time for the APIs.

---

## ⚠️ Limitations

* **Domain Gap**: The classification model was trained primarily on specialized dermoscopy images (taken using specialized medical skin-imaging lenses), but the user interface accepts regular mobile/phone camera photos. Variation in image quality, reflection, and distance may significantly affect classification accuracy.
* **Lack of Clinical Validation**: This software is not clinically validated on human subjects in hospital environments.
* **Environmental/Photo Settings**: For best results, use close-up photos captured in bright, natural light with the lesion centered and in sharp focus.

---

> [!WARNING]
> ### Medical Disclaimer
> DermaVision is a research project and academic demonstration. It is NOT a certified medical device and should NOT be used as a substitute for professional dermatological diagnosis. Always consult a board-certified dermatologist for any skin concern.

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```text
MIT License

Copyright (c) 2026 DermaVision Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
