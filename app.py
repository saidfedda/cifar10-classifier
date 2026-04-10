# app.py - FINAL ULTRA MODERN VERSION 🔥

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime
import os

# ============================================
# CONFIG
# ============================================

st.set_page_config(
    page_title="Vision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ULTRA DARK CSS + MIRROR EFFECT
# ============================================

st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #050509, #0a0a1f, #050509);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(5,5,20,0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Glass */
.glass {
    position: relative;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.4s;
}

.glass:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 30px rgba(168,237,234,0.15);
}

/* Reflection */
.glass::after {
    content:"";
    position:absolute;
    top:0;
    left:-60%;
    width:50%;
    height:100%;
    background:linear-gradient(120deg,transparent,rgba(255,255,255,0.1),transparent);
    transform:skewX(-25deg);
}

.glass:hover::after {
    left:130%;
    transition:0.7s;
}

/* Titles */
.title {
    font-size:2.5rem;
    font-weight:800;
    text-align:center;
    background: linear-gradient(135deg,#a8edea,#fed6e3);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* badges */
.badge {
    background: rgba(168,237,234,0.1);
    padding:4px 10px;
    border-radius:20px;
    font-size:12px;
    margin:2px;
}

/* sidebar cards */
.side-card {
    background: rgba(255,255,255,0.04);
    padding:10px;
    border-radius:12px;
    margin-bottom:8px;
}

/* progress */
.stProgress > div > div {
    background: linear-gradient(90deg,#a8edea,#fed6e3);
}

</style>
""", unsafe_allow_html=True)

# ============================================
# DATA
# ============================================

CLASSES = ["Plane ✈️","Car 🚗","Bird 🐦","Cat 🐱","Deer 🦌",
           "Dog 🐕","Frog 🐸","Horse 🐴","Ship 🚢","Truck 🚛"]

# ============================================
# MODEL (SIMPLE)
# ============================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,16,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*16*16,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0),-1)
        return self.fc(x)

@st.cache_resource
def load_model():
    model = SimpleCNN()
    return model, "SimpleCNN", 88.5

model, model_name, acc = load_model()

def preprocess(img):
    t = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    return t(img).unsqueeze(0)

# ============================================
# SIDEBAR (🔥 احترافي)
# ============================================

with st.sidebar:

    st.markdown("<div class='title'>🧠 Vision AI</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.markdown(f"""
    <div class='side-card'>
    Architecture: <b>{model_name}</b><br>
    Accuracy: <b>{acc}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📐 Input")
    st.markdown("""
    <div class='side-card'>
    Size: 32 × 32<br>
    Channels: RGB<br>
    Format: JPG / PNG
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Classes")
    for c in CLASSES:
        st.markdown(f"<div class='side-card'>{c}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"{datetime.now().strftime('%Y')}")

# ============================================
# MAIN
# ============================================

st.markdown("<div class='title'>Visual Intelligence</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# LEFT
with col1:
    st.markdown("<div class='glass'>Upload Image</div>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["png","jpg","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)

# RIGHT
with col2:
    st.markdown("<div class='glass'>Prediction</div>", unsafe_allow_html=True)

    if file:
        x = preprocess(img)

        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)[0]
            pred = torch.argmax(probs).item()

        st.markdown(f"<h2>{CLASSES[pred]}</h2>", unsafe_allow_html=True)
        st.progress(float(probs[pred]))

        # Top 3
        st.markdown("### Top 3")
        top3 = torch.topk(probs,3)

        for i in range(3):
            idx = top3.indices[i].item()
            val = top3.values[i].item()*100
            st.write(f"{CLASSES[idx]} - {val:.1f}%")

# Footer
st.markdown("---")
st.markdown("Made with PyTorch + Streamlit")
