import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import time
from io import BytesIO

# Adjust path to import backend components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from model import load_model, CLASSES, generate_gradcam
from utils import check_image_blur, preprocess_image, is_dermoscopy_image

# Page Configuration
st.set_page_config(
    page_title="DermaVision - Skin Lesion Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling
st.markdown("""
<style>
    /* Main Layout Adjustments */
    .reportview-container {
        background: #f8fafc;
    }
    
    /* Branding Title */
    .brand-title {
        color: #0d9488;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    
    .brand-tagline {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .custom-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        text-align: center;
        font-size: 1rem;
        border: 1px solid;
    }
    .risk-low {
        background-color: #ecfdf5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    .risk-moderate {
        background-color: #fffbeb;
        color: #92400e;
        border-color: #fde68a;
    }
    .risk-high {
        background-color: #fef2f2;
        color: #991b1b;
        border-color: #fecaca;
    }
    
    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #f1f5f9;
        border-left: 5px solid #64748b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #475569;
    }
    
    /* Custom Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def get_cached_model():
    weights_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model.pth")
    if not os.path.exists(weights_path):
        from huggingface_hub import hf_hub_download
        print("Downloading model.pth from Hugging Face...")
        try:
            # Download to the backend directory where it's expected
            backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
            hf_hub_download(repo_id="MF180/DermaVision-EfficientNet", filename="model.pth", local_dir=backend_dir)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
    return load_model(weights_path)

model = get_cached_model()

# Initialize session state for scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

def add_to_history(name, risk_level, confidence):
    # Keep last 5 scans
    scan = {
        "timestamp": time.strftime("%H:%M:%S"),
        "label": name,
        "risk": risk_level,
        "confidence": confidence
    }
    st.session_state.scan_history.insert(0, scan)
    if len(st.session_state.scan_history) > 5:
        st.session_state.scan_history = st.session_state.scan_history[:5]

# Sidebar - Brand & History Panel
with st.sidebar:
    st.markdown("<h2 style='color:#0d9488; margin-bottom:5px;'>🩺 DermaVision</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-style:italic;'>Early detection saves lives</p>", unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("### 🕒 Recent Scans (Last 5)")
    if not st.session_state.scan_history:
        st.info("No recent scans yet.")
    else:
        for idx, scan in enumerate(st.session_state.scan_history):
            color = "#059669" if scan["risk"] == "Low" else "#d97706" if scan["risk"] == "Moderate" else "#dc2626"
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 8px; background: #fafafa;">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8;">
                        <span>Scan #{len(st.session_state.scan_history)-idx}</span>
                        <span>{scan['timestamp']}</span>
                    </div>
                    <div style="font-weight: bold; margin-top:2px; font-size:0.95rem;">{scan['label']}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:5px;">
                        <span style="font-size:0.8rem; padding: 2px 8px; border-radius:10px; background-color:{color}22; color:{color}; font-weight:600;">{scan['risk']} Risk</span>
                        <span style="font-weight:600; font-size:0.9rem; color:#475569;">{scan['confidence']*100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.write("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8;">
            Version 1.0.0 (FastAPI Backend + Streamlit Dashboard)<br>
            Powered by PyTorch & EfficientNetV2-S
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Application Layout
st.markdown("<h1 class='brand-title'>DermaVision Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-tagline'>Clinical-grade deep learning classification & lesion visualization</p>", unsafe_allow_html=True)

# Tabs
tab_scan, tab_about = st.tabs(["🩺 Lesion Scan", "📊 About the Model"])

with tab_scan:
    col_input, col_results = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 📷 Upload Dermoscopy Image")
        st.write("Select a high-quality, close-up photograph of the skin lesion. Supported formats: JPG, JPEG, PNG.")
        
        # Mandatory Consent Checkbox
        is_confirmed = st.checkbox("I confirm this is a close-up photograph of a skin lesion.")
        
        uploaded_file = None
        if is_confirmed:
            uploaded_file = st.file_uploader(
                "Choose a file...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
        else:
            st.info("Please check the confirmation box above to enable uploads.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Read bytes
            image_bytes = uploaded_file.read()
            
            # Show original uploaded image preview
            try:
                pil_preview = Image.open(BytesIO(image_bytes))
                st.image(pil_preview, caption="Uploaded Image Preview", use_container_width=True)
            except Exception as e:
                st.error("Error displaying image preview: " + str(e))
                
    with col_results:
        if uploaded_file is not None:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 Diagnostic Report")
            
            # Step 1: Blur Check
            is_blurry, variance, err = check_image_blur(image_bytes, threshold=50.0)
            
            # Step 1.5: Out-of-Distribution / Non-Skin Check
            is_valid_skin, skin_err = is_dermoscopy_image(image_bytes)
            
            if err:
                st.error(f"Image analysis error: {err}")
            elif is_blurry:
                st.markdown(
                    f"""
                    <div style="background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 15px; border-radius: 8px; margin-bottom:15px;">
                        <strong>⚠️ Image Too Blurry (Variance: {variance:.1f})</strong><br>
                        The uploaded image fails our blur detection threshold (50.0). Please ensure the camera is in focus, close to the lesion, and has sufficient illumination.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif not is_valid_skin:
                st.markdown(
                    f"""
                    <div style="background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 15px; border-radius: 8px; margin-bottom:15px;">
                        <strong>⚠️ Invalid Image Detected</strong><br>
                        {skin_err}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Step 2: Prediction Inference
                with st.spinner("Analysing your image..."):
                    # Preprocess
                    input_tensor, original_img_np = preprocess_image(image_bytes)
                    
                    # Run model
                    import torch
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                    
                    confidence, class_idx = torch.max(probabilities, dim=0)
                    confidence = float(confidence.item())
                    class_idx = int(class_idx.item())
                    
                    # Map to class details
                    pred_class = CLASSES[class_idx]
                    
                    # Generate Grad-CAM Heatmap
                    with torch.enable_grad():
                        gradcam_base64 = generate_gradcam(model, input_tensor, original_img_np, class_idx)
                        
                    # Save scan history
                    add_to_history(pred_class["name"], pred_class["risk_level"], confidence)
                
                # Display Results
                risk_style = "risk-low" if pred_class['risk_level'] == "Low" else "risk-moderate" if pred_class['risk_level'] == "Moderate" else "risk-high"
                
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin-bottom: 5px; color:#475569;">Detected Lesion Class:</h4>
                        <h2 style="margin-top: 0px; color:#1e293b;">{pred_class['name']}</h2>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Risk level badge
                st.markdown(
                    f"""
                    <div style="margin-bottom: 25px;">
                        <span class="risk-badge {risk_style}">{pred_class['risk_level']} Risk Level</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Confidence indicator
                st.markdown(f"**Confidence Score:** {confidence * 100:.1f}%")
                st.progress(confidence)
                
                st.markdown(
                    f"""
                    <div style="margin-top: 20px; font-size: 0.95rem; color:#475569; line-height:1.5;">
                        <strong>Description:</strong> {pred_class['desc']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display Side-by-Side Images (Original vs Heatmap)
                st.markdown("### 🗺️ Grad-CAM Visual Heatmap")
                st.write("Heatmap highlights (in red/yellow) the specific region of the image the neural network focused on to make this prediction.")
                
                col_orig, col_cam = st.columns(2)
                with col_orig:
                    st.image(pil_preview, caption="Original Image", use_container_width=True)
                with col_cam:
                    # Decode base64 gradcam back to image
                    import base64
                    from io import BytesIO
                    gc_bytes = base64.b64decode(gradcam_base64)
                    gc_pil = Image.open(BytesIO(gc_bytes))
                    st.image(gc_pil, caption="Grad-CAM Hotspot Analysis", use_container_width=True)
                    
                st.write("---")
                # Generate PDF Report
                try:
                    from fpdf import FPDF
                    import tempfile
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 20)
                    pdf.cell(0, 15, "DermaVision Diagnostic Report", new_x="LMARGIN", new_y="NEXT", align="C")
                    pdf.ln(5)
                    
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 8, f"Detected Lesion: {pred_class['name']}", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 12)
                    pdf.cell(0, 8, f"Risk Level: {pred_class['risk_level']}", new_x="LMARGIN", new_y="NEXT")
                    pdf.cell(0, 8, f"Confidence Score: {confidence*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    pdf.multi_cell(0, 6, f"Description: {pred_class['desc']}")
                    pdf.ln(10)
                    
                    # Save images temporarily to embed
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
                        pil_preview_rgb = pil_preview.convert('RGB')
                        pil_preview_rgb.save(tmp_orig.name, format="JPEG")
                        orig_path = tmp_orig.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_gc:
                        gc_pil_rgb = gc_pil.convert('RGB')
                        gc_pil_rgb.save(tmp_gc.name, format="JPEG")
                        gc_path = tmp_gc.name
                        
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.cell(0, 10, "Visual Analysis (Original vs Grad-CAM):", new_x="LMARGIN", new_y="NEXT")
                    y_pos = pdf.get_y()
                    pdf.image(orig_path, x=10, y=y_pos, w=85)
                    pdf.image(gc_path, x=105, y=y_pos, w=85)
                    
                    pdf_bytes = pdf.output() # In fpdf2, this returns bytearray
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=bytes(pdf_bytes),
                        file_name="dermavision_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown(
                """
                <div class="disclaimer-banner">
                    <strong>⚠️ Medical Disclaimer:</strong><br>
                    This application is a demonstration powered by a machine learning model. It does NOT provide a certified medical diagnosis. Always consult a qualified physician or dermatologist for professional medical advice.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("💡 Upload a dermoscopy photo on the left to start analysis.")

with tab_about:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("## Technical Architecture")
    st.write(
        """
        DermaVision uses a deep learning classifier to categorize dermoscopy images of skin lesions into seven classes.
        """
    )
    
    st.markdown("### 🧠 The Model: EfficientNetV2-S")
    st.write(
        """
        EfficientNetV2-S is a state-of-the-art convolutional neural network optimized for fast training speeds and parameter efficiency.
        The base network is initialized with ImageNet pre-trained feature extraction weights, and the classification classifier head
        is adapted for our target dataset classes.
        """
    )
    
    st.markdown("### 📂 Training Dataset: HAM10000")
    st.write(
        """
        The model is designed to be trained on the **HAM10000** dataset:
        - **Total Images**: 10,015 high-resolution dermoscopic images
        - **Target Classes (7)**:
            1. *Melanoma* (Malignant) - High Risk
            2. *Melanocytic Nevus* (Benign Mole) - Low Risk
            3. *Basal Cell Carcinoma* (Malignant) - High Risk
            4. *Actinic Keratosis* (Pre-cancerous) - Moderate Risk
            5. *Seborrheic Keratosis* (Benign Keratosis) - Low Risk
            6. *Dermatofibroma* (Benign Growth) - Low Risk
            7. *Vascular Lesion* (Benign Growth) - Low Risk
        """
    )
    
    st.markdown("### 📊 Metrics & Training Specifications")
    st.write(
        """
        To replicate or train the model to clinical grade:
        - **Class Imbalance Mitigation**: Utilized a `WeightedRandomSampler` to balance batch feeding and **Focal Loss** to penalize errors on minority classes.
        - **Augmentations**: Random flips (horizontal & vertical), color jitter (brightness, contrast, saturation), rotation (±30°), and random cutout.
        - **Optimization**: Cosine Annealing learning rate scheduler starting at `lr=1e-4`, running for 20 epochs.
        - **Evaluation Goals**: Best checkpoint selected by validation AUC-ROC (target > 0.90), evaluating Sensitivity (recall of malignancy), Specificity, and confusion matrices.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
