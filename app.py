import os
import json
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# -----------------------
# CONFIG
# -----------------------
MODEL_DIR = "cervical_multimodal_model"  # folder containing best_model.pt + config + scaler
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load bundle
# -----------------------
@st.cache_resource
def load_bundle():
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        cfg = json.load(f)

    mean = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy")).astype(np.float32)
    scale = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy")).astype(np.float32)

    return cfg, mean, scale

cfg, scaler_mean, scaler_scale = load_bundle()
FEATURES = cfg["FEATURE_COLS"]
IMG_SIZE = cfg["IMG_SIZE"]

# -----------------------
# Model definition (MUST match training)
# -----------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)  # no download in deployment
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)

class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class FusionClassifier(nn.Module):
    def __init__(self, tab_in_dim, n_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(out_dim=256)
        self.tab_enc = TabularEncoder(in_dim=tab_in_dim, out_dim=128)
        self.head = nn.Sequential(
            nn.Linear(256+128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, img, tab):
        a = self.img_enc(img)
        b = self.tab_enc(tab)
        x = torch.cat([a, b], dim=1)
        return self.head(x)

@st.cache_resource
def load_model():
    model = FusionClassifier(tab_in_dim=len(FEATURES), n_classes=2).to(DEVICE)
    state = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# -----------------------
# Preprocessing
# -----------------------
img_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def preprocess_tabular(values_dict):
    x = np.array([values_dict[f] for f in FEATURES], dtype=np.float32)
    x = (x - scaler_mean) / (scaler_scale + 1e-8)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

@torch.no_grad()
def predict(image: Image.Image, tab_values: dict):
    img_t = img_tfms(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    tab_t = preprocess_tabular(tab_values).to(DEVICE)

    logits = model(img_t, tab_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[1]), probs

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Cervical Cancer Detection (Multimodal)", layout="centered")
st.title("Cervical Cancer Detection Using Deep Learning (Multimodal)")
st.caption("Pap smear cytology image + key clinical risk factors → Normal vs Abnormal prediction")

uploaded = st.file_uploader("Upload a Pap smear cytology image (BMP/PNG/JPG)", type=["bmp","png","jpg","jpeg","tif","tiff"])

st.subheader("Clinical Risk Inputs (Basic)")
col1, col2 = st.columns(2)

# Create inputs matching training feature names
inputs = {}
with col1:
    inputs["Age"] = st.number_input("Age", min_value=10, max_value=100, value=26)
    inputs["Number_of_sexual_partners"] = st.number_input("Number of sexual partners", min_value=0, max_value=50, value=2)
    inputs["First_sexual_intercourse"] = st.number_input("Age at first sexual intercourse", min_value=5, max_value=60, value=17)
    inputs["Num_of_pregnancies"] = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=2)

with col2:
    inputs["Smokes"] = st.selectbox("Smokes?", [0, 1], index=0)
    inputs["STDs"] = st.selectbox("History of STDs?", [0, 1], index=0)
    inputs["Hormonal_Contraceptives"] = st.selectbox("Hormonal Contraceptives?", [0, 1], index=1)
    inputs["IUD"] = st.selectbox("IUD usage?", [0, 1], index=0)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Pap smear image", use_container_width=True)

    if st.button("Predict"):
        pred, abnormal_prob, probs = predict(img, inputs)

        if pred == 1:
            st.error(f"Prediction: ABNORMAL (Cancer-risk cytology)  |  Probability: {abnormal_prob:.3f}")
        else:
            st.success(f"Prediction: NORMAL  |  Abnormal Probability: {abnormal_prob:.3f}")

        st.write("Class probabilities:", {"Normal": float(probs[0]), "Abnormal": float(probs[1])})

else:
    st.info("Upload an image to run the multimodal cervical cancer prediction.")
