"""
=============================================================================
CERVICAL CANCER MULTIMODAL SCREENING SUPPORT TOOL
=============================================================================
Author: Nokutenda Bridget Chibisa
University of Zimbabwe - Department of Analytics and Informatics

A multimodal deep learning prototype combining Pap smear cytology images
and clinical risk factors for cervical cancer screening support.

FEATURES:
- Multimodal prediction (image + clinical data)
- Grad-CAM image explainability
- SHAP clinical risk factor explainability
- Professional medical screening UI

DISCLAIMER: This is a research prototype and NOT a clinical diagnostic tool.
=============================================================================
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- Optional SHAP ----
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =============================================================================
# MODEL DEFINITIONS (must match training exactly)
# =============================================================================

class ImageBranch(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate))
    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features.view(features.size(0), -1))


class TabularBranch(nn.Module):
    def __init__(self, input_dim=8, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout_rate),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout_rate))
    def forward(self, x):
        return self.network(x)


class MultimodalModel(nn.Module):
    def __init__(self, num_features=8, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.image_branch = ImageBranch(dropout_rate)
        self.tabular_branch = TabularBranch(num_features, dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes))
    def forward(self, image, tabular):
        img_features = self.image_branch(image)
        tab_features = self.tabular_branch(tabular)
        fused = torch.cat([img_features, tab_features], dim=1)
        return self.classifier(fused)


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)
    
    def _fwd_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image, tabular, target_class=None):
        self.model.eval()
        output = self.model(image, tabular)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = torch.nn.functional.interpolate(
            cam, size=(image.shape[2], image.shape[3]),
            mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()


# =============================================================================
# SHAP WRAPPER
# =============================================================================

class TabularPathWrapper(nn.Module):
    def __init__(self, multimodal_model, ref_img_features):
        super().__init__()
        self.tabular_branch = multimodal_model.tabular_branch
        self.classifier = multimodal_model.classifier
        self.register_buffer('ref_img_features', ref_img_features)
    
    def forward(self, tabular_input):
        tab_features = self.tabular_branch(tabular_input)
        img_features = self.ref_img_features.expand(tabular_input.size(0), -1)
        fused = torch.cat([img_features, tab_features], dim=1)
        return torch.softmax(self.classifier(fused), dim=1)


# =============================================================================
# LOAD MODEL & ASSETS
# =============================================================================

@st.cache_resource
def load_model():
    """Load model and preprocessing assets."""
    model_dir = 'cervical_multimodal_model'
    device = torch.device('cpu')
    
    # Load config
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load model
    model = MultimodalModel(
        num_features=config['num_features'],
        num_classes=config['num_classes'],
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device)
    )
    model.eval()
    
    # Load scaler
    scaler_mean = np.load(os.path.join(model_dir, 'scaler_mean.npy'))
    scaler_scale = np.load(os.path.join(model_dir, 'scaler_scale.npy'))
    
    # Load SHAP background (if available)
    shap_bg_path = os.path.join(model_dir, 'shap_background.npy')
    shap_background = None
    if os.path.exists(shap_bg_path):
        shap_background = np.load(shap_bg_path)
    
    return model, config, scaler_mean, scaler_scale, shap_background, device


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model input."""
    img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0), img


def preprocess_tabular(features_dict, scaler_mean, scaler_scale, feature_names):
    """Scale clinical features using saved scaler parameters."""
    values = np.array([features_dict[f] for f in feature_names], dtype=np.float32)
    scaled = (values - scaler_mean) / scaler_scale
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0), values


def make_prediction(model, image_tensor, tabular_tensor, device):
    """Run multimodal prediction."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device), tabular_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence, probs[0].cpu().numpy()


def generate_gradcam(model, image_tensor, tabular_tensor, device):
    """Generate Grad-CAM heatmap."""
    target_layer = model.image_branch.backbone[7][1].conv2
    gradcam = GradCAM(model, target_layer)
    
    image_tensor.requires_grad_(True)
    heatmap = gradcam.generate(
        image_tensor.to(device), tabular_tensor.to(device), target_class=1
    )
    return heatmap


def compute_live_shap(model, tabular_tensor, shap_background, device):
    """Compute SHAP values for a single live prediction."""
    if not SHAP_AVAILABLE or shap_background is None:
        return None
    
    try:
        # Get reference image features from background
        bg_tensor = torch.tensor(shap_background[:50], dtype=torch.float32).to(device)
        
        # Create a simple reference (mean image features)
        with torch.no_grad():
            ref_img = torch.zeros(1, 256).to(device)
        
        wrapper = TabularPathWrapper(model, ref_img).to(device)
        wrapper.eval()
        
        # Use GradientExplainer (more compatible with this model)
        explainer = shap.GradientExplainer(wrapper, bg_tensor)
        shap_vals = explainer.shap_values(tabular_tensor.to(device))
        
        # Handle different output shapes
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            sv = shap_vals[1][0]  # Abnormal class, single sample
        else:
            sv = shap_vals[0]
        
        # GradientExplainer may return (n_features, n_classes) — take abnormal column
        sv = np.array(sv)
        if sv.ndim == 2:
            sv = sv[:, 1]  # Take abnormal class column
        
        return sv
    except Exception as e:
        st.warning(f"SHAP computation unavailable: {e}")
        return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_gradcam_figure(original_img, heatmap):
    """Create Grad-CAM overlay figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    axes[0].imshow(original_img.resize((224, 224)))
    axes[0].set_title('Uploaded Pap Smear', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(original_img.resize((224, 224)))
    axes[1].imshow(heatmap, cmap='jet', alpha=0.45)
    axes[1].set_title('Grad-CAM: Model Attention', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def create_shap_figure(shap_values, feature_names, feature_values):
    """Create SHAP horizontal bar chart."""
    if shap_values is None:
        return None
    
    sv = shap_values.cpu().detach().numpy() if hasattr(shap_values, 'cpu') else np.array(shap_values)
    
    sort_idx = np.argsort(np.abs(sv))
    sorted_sv = sv[sort_idx]
    sorted_names = [f"{feature_names[i]} = {feature_values[i]:.1f}" for i in sort_idx]
    colors = ['#E74C3C' if v > 0 else '#2E86C1' for v in sorted_sv]
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(range(len(sorted_sv)), sorted_sv, color=colors,
            edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('SHAP Value (contribution to prediction)', fontsize=10)
    ax.set_title('Clinical Risk Factor Contributions', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='gray', linewidth=0.8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='→ Abnormal'),
        Patch(facecolor='#2E86C1', label='→ Normal')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
             framealpha=0.9)
    ax.grid(True, axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Cervical Cancer Screening Support",
        page_icon="🔬",
        layout="wide"
    )
    
    # ---- Custom CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        color: #555;
    }
    
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    .result-abnormal {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    .result-box h2 {
        margin: 0;
        font-size: 1.6rem;
    }
    .result-box p {
        margin: 0.3rem 0 0;
        font-size: 1rem;
    }
    
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #856404;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stSidebar"] {
        background: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ---- Header ----
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Cervical Cancer Screening Support Tool</h1>
        <p>Multimodal Deep Learning: Pap Smear Cytology + Clinical Risk Factors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Research Prototype Disclaimer:</strong> This tool is a university capstone project 
        and is NOT a medical diagnostic device. Predictions are for educational and research purposes only. 
        Always consult a qualified healthcare professional for cervical cancer screening.
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Load Model ----
    try:
        model, config, scaler_mean, scaler_scale, shap_background, device = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    feature_names = config['feature_names']
    
    # ---- Sidebar: Clinical Risk Factors ----
    st.sidebar.header("📋 Patient Clinical Information")
    st.sidebar.markdown("Enter the patient's clinical risk factors below.")
    
    features_dict = {}
    
    features_dict['Age'] = st.sidebar.number_input(
        "Age (years)", min_value=13, max_value=85, value=30, step=1,
        help="Patient's current age"
    )
    features_dict['Number of sexual partners'] = st.sidebar.number_input(
        "Number of sexual partners", min_value=0, max_value=30, value=2, step=1
    )
    features_dict['First sexual intercourse'] = st.sidebar.number_input(
        "Age at first sexual intercourse", min_value=10, max_value=40, value=17, step=1
    )
    features_dict['Num of pregnancies'] = st.sidebar.number_input(
        "Number of pregnancies", min_value=0, max_value=15, value=1, step=1
    )
    features_dict['Smokes'] = st.sidebar.selectbox(
        "Smokes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    features_dict['STDs'] = st.sidebar.selectbox(
        "History of STDs", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    features_dict['Hormonal Contraceptives'] = st.sidebar.selectbox(
        "Uses hormonal contraceptives", options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    features_dict['IUD'] = st.sidebar.selectbox(
        "Uses IUD", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    # ---- Main Area: Image Upload ----
    st.markdown("### 📤 Upload Pap Smear Cytology Image")
    
    uploaded_file = st.file_uploader(
        "Upload a Pap smear cell image (.jpg, .png, .bmp)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a single-cell Pap smear cytology image for analysis"
    )
    
    if uploaded_file is not None:
        # Show uploaded image
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption="Uploaded Image", width=280)
        with col_info:
            st.markdown("""
            <div class="info-box">
                <strong>ℹ️ How it works:</strong><br>
                The system analyses the cell image using a ResNet18 deep learning model 
                and combines it with the clinical risk factors you entered to make a 
                multimodal prediction.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Clinical inputs summary:**
            - Age: {features_dict['Age']} | Partners: {features_dict['Number of sexual partners']}
            - Smokes: {'Yes' if features_dict['Smokes'] else 'No'} | STDs: {'Yes' if features_dict['STDs'] else 'No'}
            - Contraceptives: {'Yes' if features_dict['Hormonal Contraceptives'] else 'No'} | IUD: {'Yes' if features_dict['IUD'] else 'No'}
            """)
    
    # ---- Predict Button ----
    if uploaded_file is not None:
        if st.button("🔍 Analyse", type="primary", use_container_width=True):
            
            with st.spinner("Running multimodal analysis..."):
                # Preprocess
                image_tensor, original_img = preprocess_image(uploaded_file)
                tabular_tensor, raw_values = preprocess_tabular(
                    features_dict, scaler_mean, scaler_scale, feature_names
                )
                
                # Predict
                pred_class, confidence, probs = make_prediction(
                    model, image_tensor, tabular_tensor, device
                )
                
                prediction_label = config['class_names'][pred_class]
                prob_normal = probs[0]
                prob_abnormal = probs[1]
            
            # ---- Display Results ----
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            if prediction_label == 'Abnormal':
                st.markdown(f"""
                <div class="result-box result-abnormal">
                    <h2>🔴 ABNORMAL</h2>
                    <p>Confidence: {confidence:.1%} | P(Normal): {prob_normal:.4f} | P(Abnormal): {prob_abnormal:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-normal">
                    <h2>🟢 NORMAL</h2>
                    <p>Confidence: {confidence:.1%} | P(Normal): {prob_normal:.4f} | P(Abnormal): {prob_abnormal:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ---- Explainability Section ----
            st.markdown("---")
            st.markdown("### 🔍 Explainability: Why This Prediction?")
            
            exp_col1, exp_col2 = st.columns(2)
            
            # Grad-CAM
            with exp_col1:
                st.markdown("**🖼️ Image Explainability (Grad-CAM)**")
                st.caption("Highlighted regions show where the model focused in the cell image.")
                try:
                    heatmap = generate_gradcam(model, image_tensor, tabular_tensor, device)
                    gradcam_fig = create_gradcam_figure(original_img, heatmap)
                    st.pyplot(gradcam_fig)
                    plt.close(gradcam_fig)
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed: {e}")
            
            # SHAP
            with exp_col2:
                st.markdown("**📊 Clinical Factor Explainability (SHAP)**")
                st.caption("Bar length shows how much each factor influenced the prediction.")
                
                if SHAP_AVAILABLE and shap_background is not None:
                    try:
                        shap_vals = compute_live_shap(
                            model, tabular_tensor, shap_background, device
                        )
                        shap_fig = create_shap_figure(shap_vals, feature_names, raw_values)
                        if shap_fig:
                            st.pyplot(shap_fig)
                            plt.close(shap_fig)
                        else:
                            st.info("SHAP values could not be computed for this prediction.")
                    except Exception as e:
                        st.info(f"SHAP not available: {e}")
                else:
                    st.info(
                        "SHAP explainability requires the `shap` package and background data. "
                        "The Grad-CAM visualization on the left still shows image-based explainability."
                    )
            
            # ---- Interpretation Guide ----
            st.markdown("---")
            with st.expander("📖 How to interpret these results"):
                st.markdown("""
                **Prediction:** The model combines both the cell image and your clinical inputs 
                to classify the sample as Normal or Abnormal.
                
                **Grad-CAM (left):** The heatmap shows which regions of the cell image the model 
                focused on. Warm colours (red/yellow) indicate high attention areas. In abnormal 
                predictions, the model should focus on irregular cell structures such as abnormal 
                nuclei or cytoplasmic changes.
                
                **SHAP (right):** Each bar shows how much a clinical factor contributed to the 
                prediction. Red bars push toward Abnormal, blue bars push toward Normal. Longer 
                bars mean stronger influence. For example, a long red bar for "STDs = 1" means 
                having an STD history significantly increased the abnormal risk score.
                
                **Important:** This is a screening support tool, not a diagnosis. High-confidence 
                abnormal predictions should prompt further clinical investigation, not replace it.
                """)
    
    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
        "Cervical Cancer Multimodal Screening Support Tool | "
        "Nokutenda Bridget Chibisa | University of Zimbabwe | 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
