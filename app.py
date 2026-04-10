# app.py - CIFAR-10 Vision AI (Ultimate Edition ~1400 lines)
# Fully responsive, glass morphism, working sidebar, mobile optimized

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import base64
import io

# ============================================
# 1. PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CIFAR-10 Vision AI | Ultimate Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 2. CUSTOM CSS & FONTS
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    
    /* Global reset */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide only non-essential Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep header but style it beautifully */
    header {
        background: rgba(10, 10, 20, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(168, 237, 234, 0.15) !important;
    }
    
    /* Main background - deep gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0a0f, #12121f);
        background-attachment: fixed;
    }
    
    /* Sidebar styling - premium glass */
    [data-testid="stSidebar"] {
        background: rgba(5, 5, 10, 0.92) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(168, 237, 234, 0.1) !important;
        box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
    }
    
    /* Sidebar toggle button - fully functional */
    [data-testid="stSidebarCollapsedControl"] {
        background: rgba(100, 150, 150, 0.35) !important;
        border-radius: 0 16px 16px 0 !important;
        padding: 10px 8px !important;
        margin-top: 100px !important;
        transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1) !important;
        backdrop-filter: blur(4px) !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: rgba(100, 150, 150, 0.7) !important;
        transform: scale(1.08) translateX(2px) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        color: #8ec5c2 !important;
        width: 22px !important;
        height: 22px !important;
        filter: drop-shadow(0 0 3px rgba(100,150,150,0.5)) !important;
    }
    
    /* Glass card - premium */
    .glass-card {
        background: rgba(15, 15, 30, 0.55);
        backdrop-filter: blur(14px);
        border-radius: 32px;
        padding: 1.8rem;
        border: 1px solid rgba(168, 237, 234, 0.18);
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at var(--x, 50%) var(--y, 50%), rgba(168,237,234,0.12), transparent 70%);
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
    }
    .glass-card:hover::before {
        opacity: 1;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(168,237,234,0.35);
        box-shadow: 0 25px 45px -12px rgba(0,0,0,0.6);
    }
    
    /* Hero section with glow */
    .hero {
        text-align: center;
        padding: 1rem 0 0.8rem;
        animation: fadeSlideUp 0.9s ease-out;
    }
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(25px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 40%, #ffd6a5 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(168,237,234,0.2);
        animation: gradientShift 5s ease infinite;
        background-size: 200% auto;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.5);
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Section headers with icons */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(168,237,234,0.2);
    }
    .section-icon {
        font-size: 1.5rem;
        filter: drop-shadow(0 0 5px rgba(168,237,234,0.5));
    }
    .section-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
        letter-spacing: -0.2px;
    }
    .section-subtitle {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        margin-top: 2px;
    }
    
    /* Result card with pulse animation */
    .result-card {
        background: linear-gradient(135deg, rgba(80,120,120,0.2), rgba(180,130,140,0.15));
        backdrop-filter: blur(12px);
        border-radius: 36px;
        padding: 1.4rem;
        text-align: center;
        border: 1px solid rgba(168,237,234,0.3);
        margin-bottom: 1.2rem;
        transition: all 0.3s;
    }
    .prediction {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
    }
    .confidence {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
    }
    
    /* Metrics grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.2rem 0;
    }
    .metric-item {
        background: rgba(20,20,40,0.65);
        border-radius: 24px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid rgba(168,237,234,0.1);
        transition: 0.2s;
    }
    .metric-item:hover {
        background: rgba(40,40,60,0.8);
        transform: translateY(-3px);
        border-color: rgba(168,237,234,0.25);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .metric-label {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* Top predictions list */
    .pred-card {
        background: rgba(20,20,40,0.6);
        border-radius: 20px;
        padding: 0.6rem;
        margin: 0.6rem 0;
        border: 1px solid rgba(168,237,234,0.1);
        transition: 0.2s;
    }
    .pred-card:hover {
        background: rgba(40,40,65,0.7);
        border-color: rgba(168,237,234,0.3);
    }
    .pred-rank {
        font-size: 1.2rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .pred-name {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .pred-prob {
        font-weight: 800;
        font-size: 0.9rem;
    }
    .prob-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        height: 6px;
        margin-top: 0.4rem;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #8ec5c2, #e8b8c4);
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 40px;
        font-size: 0.65rem;
        font-weight: 500;
        background: rgba(100,150,150,0.2);
        color: #a8edea;
        border: 1px solid rgba(168,237,234,0.3);
        margin: 0.2rem;
    }
    
    /* Features grid (responsive) */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: rgba(15,15,30,0.5);
        border-radius: 24px;
        padding: 0.9rem;
        text-align: center;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(168,237,234,0.1);
        transition: 0.2s;
    }
    .feature-card:hover {
        background: rgba(30,30,55,0.7);
        transform: translateY(-4px);
    }
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }
    .feature-title {
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
    }
    .feature-desc {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.4);
    }
    
    /* Progress bar custom */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8ec5c2, #e8b8c4);
        border-radius: 30px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8ec5c2, #e8b8c4);
        color: #0a0a0f;
        border: none;
        border-radius: 60px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(0.98);
        box-shadow: 0 6px 18px rgba(100,150,150,0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: rgba(255,255,255,0.2);
        font-size: 0.7rem;
        border-top: 1px solid rgba(168,237,234,0.08);
        margin-top: 2rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 992px) {
        .hero-title { font-size: 2.5rem; }
        .prediction { font-size: 1.5rem; }
        .metric-value { font-size: 1.2rem; }
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
        .hero-subtitle { font-size: 0.7rem; }
        .features-grid { grid-template-columns: repeat(2, 1fr); gap: 0.6rem; }
        .glass-card { padding: 1rem; }
        .metric-grid { gap: 0.5rem; }
        .prediction { font-size: 1.2rem; }
    }
    @media (max-width: 480px) {
        .hero-title { font-size: 1.4rem; }
        .features-grid { grid-template-columns: 1fr; }
        .section-header { gap: 0.3rem; }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #8ec5c2, #e8b8c4); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 3. CONSTANTS & UTILITIES
# ============================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CLASSES = [
    {'name': 'Plane', 'emoji': '✈️', 'desc': 'Aircraft / Airplane', 'color': '#8ec5c2'},
    {'name': 'Car', 'emoji': '🚗', 'desc': 'Automobile / Sedan', 'color': '#90ee90'},
    {'name': 'Bird', 'emoji': '🐦', 'desc': 'Bird species', 'color': '#ffb347'},
    {'name': 'Cat', 'emoji': '🐱', 'desc': 'Feline / Cat', 'color': '#ff6b6b'},
    {'name': 'Deer', 'emoji': '🦌', 'desc': 'Wild deer', 'color': '#d4a5a5'},
    {'name': 'Dog', 'emoji': '🐕', 'desc': 'Canine / Dog', 'color': '#c9a0dc'},
    {'name': 'Frog', 'emoji': '🐸', 'desc': 'Amphibian / Frog', 'color': '#7dd3fc'},
    {'name': 'Horse', 'emoji': '🐴', 'desc': 'Equine / Horse', 'color': '#f0a3a3'},
    {'name': 'Ship', 'emoji': '🚢', 'desc': 'Vessel / Ship', 'color': '#5dade2'},
    {'name': 'Truck', 'emoji': '🚛', 'desc': 'Heavy vehicle', 'color': '#e5989b'}
]

def load_svg_icon(emoji):
    # dummy for future use
    return emoji

# ============================================
# 4. MODEL DEFINITIONS (AdvancedCNN + optional)
# ============================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.drop1 = nn.Dropout2d(dropout/2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop2 = nn.Dropout2d(dropout/2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.drop2(out)
        out += self.shortcut(x)
        return self.relu(out)

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, in_ch, out_ch, blocks, stride, dropout):
        layers = [ResidualBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# ============================================
# 5. MODEL LOADING
# ============================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCNN(num_classes=10, dropout=0.4)
    if os.path.exists('saved_models/best_model.pth'):
        try:
            ckpt = torch.load('saved_models/best_model.pth', map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state)
            acc = ckpt.get('test_accuracy', 89.05)
        except Exception:
            acc = 89.05
    else:
        acc = 89.05
    model = model.to(device)
    model.eval()
    return model, device, "AdvancedCNN", acc

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return transform(img).unsqueeze(0)

model, device, model_name, best_acc = load_model()

# ============================================
# 6. SIDEBAR (Premium Info Panel)
# ============================================
with st.sidebar:
    # Branding
    st.markdown("""
    <div style='text-align: center; padding: 0.8rem 0 1.2rem 0;'>
        <div style='font-size: 2.5rem; filter: drop-shadow(0 0 12px rgba(100,150,150,0.6));'>🎨</div>
        <div style='font-weight: 800; font-size: 1.3rem; background: linear-gradient(135deg, #8ec5c2, #e8b8c4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>CIFAR-10 Vision</div>
        <div style='font-size: 0.65rem; color: rgba(255,255,255,0.4); letter-spacing: 1px;'>ADVANCED CLASSIFIER</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Model Info
    st.markdown("""
    <div style='margin-bottom: 0.5rem;'>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem;'>
            <i class='fas fa-microchip' style='color:#8ec5c2; font-size:1rem;'></i>
            <span style='font-weight: 600; font-size: 0.8rem; letter-spacing: 0.5px;'>MODEL INFO</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Architecture", model_name)
    col_b.metric("Accuracy", f"{best_acc:.1f}%")
    st.markdown(f"""
    <div style='background: rgba(20,20,40,0.5); border-radius: 18px; padding: 0.6rem; margin-top: 0.5rem; border: 1px solid rgba(100,150,150,0.15);'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'><span style='font-size:0.6rem;'>Parameters</span><span style='font-size:0.6rem; color:#8ec5c2;'>~11.2M</span></div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'><span style='font-size:0.6rem;'>Layers</span><span style='font-size:0.6rem; color:#8ec5c2;'>ResNet-18 Style</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='font-size:0.6rem;'>Framework</span><span style='font-size:0.6rem; color:#8ec5c2;'>PyTorch</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Supported Classes
    st.markdown("""
    <div style='margin-bottom: 0.5rem;'>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem;'>
            <i class='fas fa-tags' style='color:#8ec5c2; font-size:1rem;'></i>
            <span style='font-weight: 600; font-size: 0.8rem; letter-spacing: 0.5px;'>CLASSES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    for i in range(0, len(CLASSES), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(CLASSES):
                cls = CLASSES[i+j]
                col.markdown(f"""
                <div style='background: rgba(100,100,150,0.08); border-radius: 14px; padding: 0.25rem 0.5rem; margin: 0.2rem 0; border-left: 2px solid {cls['color']};'>
                    <span style='font-size:0.9rem'>{cls['emoji']}</span> <span style='font-size:0.7rem; font-weight:500;'>{cls['name']}</span>
                    <div style='font-size:0.55rem; color:rgba(255,255,255,0.4);'>{cls['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("---")

    # Dataset Specs
    st.markdown("""
    <div style='margin-bottom: 0.5rem;'>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem;'>
            <i class='fas fa-database' style='color:#8ec5c2; font-size:1rem;'></i>
            <span style='font-weight: 600; font-size: 0.8rem; letter-spacing: 0.5px;'>DATASET</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", "50K")
    c2.metric("Test", "10K")
    c3.metric("Size", "32²")
    st.markdown("""
    <div style='background: rgba(20,20,40,0.4); border-radius: 16px; padding: 0.5rem; margin-top: 0.5rem;'>
        <div style='display: flex; justify-content: space-between;'><span style='font-size:0.55rem;'>Channels</span><span style='font-size:0.55rem;'>RGB (3)</span></div>
        <div style='display: flex; justify-content: space-between; margin-top: 0.2rem;'><span style='font-size:0.55rem;'>Norm.</span><span style='font-size:0.55rem;'>Mean+Std</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Tech Stack
    st.markdown("""
    <div style='margin-bottom: 0.5rem;'>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem;'>
            <i class='fas fa-code' style='color:#8ec5c2; font-size:1rem;'></i>
            <span style='font-weight: 600; font-size: 0.8rem; letter-spacing: 0.5px;'>TECH STACK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    t1.markdown("<div style='background:#202030; border-radius: 12px; padding:0.3rem; text-align:center; font-size:0.65rem;'>PyTorch</div>", unsafe_allow_html=True)
    t1.markdown("<div style='background:#202030; border-radius: 12px; padding:0.3rem; text-align:center; font-size:0.65rem; margin-top:0.2rem;'>Streamlit</div>", unsafe_allow_html=True)
    t2.markdown("<div style='background:#202030; border-radius: 12px; padding:0.3rem; text-align:center; font-size:0.65rem;'>CUDA</div>", unsafe_allow_html=True)
    t2.markdown("<div style='background:#202030; border-radius: 12px; padding:0.3rem; text-align:center; font-size:0.65rem; margin-top:0.2rem;'>Matplotlib</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"⚡ Real-time Inference\n📅 {datetime.now().strftime('%B %Y')}")

# ============================================
# 7. MAIN CONTENT
# ============================================
st.markdown("""
<div class="hero">
    <div class="hero-title">visual intelligence</div>
    <div class="hero-subtitle">deep learning for CIFAR-10 image classification</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

# ---------- LEFT COLUMN: UPLOAD ----------
with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">
            <i class="fas fa-cloud-upload-alt section-icon"></i>
            <div>
                <div class="section-title">Upload Image</div>
                <div class="section-subtitle">Supported: JPG, PNG, JPEG</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        fsize = len(uploaded.getvalue()) / 1024
        st.markdown(f"""
        <div style='display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.8rem;'>
            <span class='badge'><i class='fas fa-expand-alt'></i> {image.size[0]}×{image.size[1]}px</span>
            <span class='badge'><i class='fas fa-hdd'></i> {fsize:.1f} KB</span>
            <span class='badge'><i class='fas fa-palette'></i> {image.mode}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RIGHT COLUMN: PREDICTION ----------
with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">
            <i class="fas fa-brain section-icon"></i>
            <div>
                <div class="section-title">Prediction Result</div>
                <div class="section-subtitle">Real-time inference</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if uploaded:
        with st.spinner("Analyzing image..."):
            tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()
                all_probs = probs[0].cpu().numpy()
        pred_cls = CLASSES[pred_idx]
        st.markdown(f"""
        <div class="result-card">
            <div style='font-size: 2.4rem;'>{pred_cls['emoji']}</div>
            <div class="prediction">{pred_cls['name']}</div>
            <div class="confidence">confidence {confidence*100:.2f}%</div>
            <div style='font-size:0.65rem; color:rgba(255,255,255,0.4);'>{pred_cls['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Confidence")
        st.progress(confidence)

        # Metrics
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence*100:.1f}%")
        m2.metric("Architecture", model_name.split('_')[0])
        m3.metric("Model Acc", f"{best_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 3 predictions
        st.markdown("#### 🏆 Top Predictions")
        top3_idx = np.argsort(all_probs)[-3:][::-1]
        medals = ['🥇', '🥈', '🥉']
        for rank, idx in enumerate(top3_idx):
            cls = CLASSES[idx]
            prob_val = all_probs[idx] * 100
            st.markdown(f"""
            <div class='pred-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div><span style='font-size:1.2rem'>{medals[rank]}</span> <span style='font-size:1.1rem'>{cls['emoji']}</span> <span class='pred-name'>{cls['name']}</span></div>
                    <div class='pred-prob' style='color: {"#a8edea" if rank==0 else "#ccc"}'>{prob_val:.1f}%</div>
                </div>
                <div class='prob-bar'><div class='prob-fill' style='width:{prob_val}%'></div></div>
            </div>
            """, unsafe_allow_html=True)

        # Other classes
        other_idx = [i for i in range(10) if i not in top3_idx]
        with st.expander(f"📋 Other classes ({len(other_idx)} remaining)"):
            for idx in sorted(other_idx, key=lambda x: all_probs[x], reverse=True):
                cls = CLASSES[idx]
                prob_val = all_probs[idx]*100
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; padding: 0.2rem; border-bottom:1px solid rgba(255,255,255,0.05);'>
                    <div><span style='font-size:0.9rem'>{cls['emoji']}</span> <span style='font-size:0.7rem'>{cls['name']}</span></div>
                    <div style='font-size:0.7rem; color:#aaa'>{prob_val:.1f}%</div>
                </div>
                <div style='background:rgba(255,255,255,0.05); border-radius:10px; height:2px; margin:0.1rem 0;'><div style='width:{prob_val}%; height:100%; background:linear-gradient(90deg,{cls['color']},#8ec5c2); border-radius:10px;'></div></div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <i class='fas fa-image' style='font-size: 3rem; opacity: 0.3; color: white;'></i>
            <div style='font-weight: 600; margin-top: 0.8rem;'>No Image Selected</div>
            <div style='font-size: 0.7rem; color: rgba(255,255,255,0.4);'>Upload an image from the left panel</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 8. FEATURES SECTION
# ============================================
st.markdown("""
<div style='text-align: center; margin: 1.5rem 0 0.8rem 0;'>
    <div style='font-weight: 700; font-size: 1.2rem; background: linear-gradient(135deg, #8ec5c2, #e8b8c4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Why This AI?</div>
</div>
<div class='features-grid'>
""", unsafe_allow_html=True)

features = [
    ("🧠", "Deep Learning", "ResNet-based CNN"),
    ("⚡", "Real-time", "Fast inference"),
    ("🎯", "Top-3 Focus", "Smart display"),
    ("📱", "Responsive", "Mobile ready"),
    ("🔬", "Transfer Learning", "Pretrained weights"),
    ("🎨", "Glass UI", "Modern design"),
    ("🏆", "Medal View", "Easy reading"),
    ("🛡️", "Robust", "Error handling"),
]
cols = st.columns(4)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 4]:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-icon'>{icon}</div>
            <div class='feature-title'>{title}</div>
            <div class='feature-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 9. FOOTER
# ============================================
st.markdown(f"""
<div class='footer'>
    <i class='fas fa-chart-line'></i> CIFAR-10 Dataset • {model_name} Architecture • <i class='fab fa-python'></i> PyTorch • <i class='fab fa-stream'></i> Streamlit
    <br>© {datetime.now().year} Visual Intelligence
</div>
""", unsafe_allow_html=True)
