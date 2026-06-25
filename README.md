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
- **Batch Sampling**: Implement a PyTorch `WeightedRandomSampler` to calculate weights based on the inverse class frequency ($1 / N_c$). This guarantees that each training batch contains a balanced representation of all 7 classes.
- **Criterion**: Use **Focal Loss** instead of standard Cross Entropy. Focal Loss dynamically down-weights well-classified examples ($p > 0.5$) and focuses back-propagation steps on hard, misclassified samples:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  *Parameters used: $\gamma = 2.0$, $\alpha$ balanced class weights.*

### 3. Data Augmentation
To prevent overfitting on lighting variations and patient skin-tones, we implement a robust image augmentation pipeline:
- **Geometry**: Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$), Random Rotation ($\pm30^\circ$).
- **Color**: Color Jitter (Brightness $\pm0.2$, Contrast $\pm0.2$, Saturation $\pm0.2$, Hue $\pm0.1$).
- **Regularization**: Coarse Dropout (Cutout) to mask random $16\times16$ patches, forcing the model to look at multiple features of a lesion rather than a single focal hotspot.

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

| Metric       | Score  |
|--------------|--------|
| AUC-ROC      | > 0.90 |
| Accuracy     | TBD    |
| Sensitivity  | TBD    |
| Specificity  | TBD    |

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
