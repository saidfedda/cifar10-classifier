# app.py - واجهة احترافية فاخرة

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CIFAR-10 Vision AI | Advanced Image Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - تصميم خرافي
# ============================================

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass morphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        animation: fadeIn 1s ease-in;
    }
    
    /* Result box animation */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .result-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload box */
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.9);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DEFINITIONS
# ============================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('Plane ✈️', 'Car 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌', 
           'Dog 🐕', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚')
CLASSES_SIMPLE = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ANN Model
class ANN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# CNN Model
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate/2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate/2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# AdvancedCNN (ResNet-style)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate/2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate/2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_rate))
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
        x = self.fc(x)
        return x

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('saved_models/results_summary.json', 'r') as f:
        summary = json.load(f)
    
    best_model_name = summary['best_model']
    
    if 'ANN' in best_model_name:
        model = ANN(dropout_rate=0.3)
    elif 'CNN' in best_model_name:
        model = CNN(dropout_rate=0.4)
    else:
        model = AdvancedCNN(num_classes=10, dropout_rate=0.4)
    
    checkpoint = torch.load('saved_models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device, best_model_name, summary['best_accuracy']

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return transform(image).unsqueeze(0)

# ============================================
# SIDEBAR - معلومات واحصائيات
# ============================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="color: white; margin: 0;">🎯 CIFAR-10 AI</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">Advanced Image Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model info
    model, device, model_name, best_acc = load_model()
    
    st.markdown("### 🤖 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Architecture", model_name.split('_')[0])
    with col2:
        st.metric("Accuracy", f"{best_acc:.1f}%")
    
    st.markdown("---")
    
    # Class list
    st.markdown("### 📋 Available Classes")
    for cls in CLASSES:
        st.markdown(f"- {cls}")
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Dataset Statistics")
    st.metric("Total Classes", "10")
    st.metric("Image Size", "32x32 pixels")
    st.metric("Training Samples", "50,000")
    st.metric("Test Samples", "10,000")
    
    st.markdown("---")
    
    # Footer in sidebar
    st.caption(f"🚀 Powered by PyTorch\n✨ Deployed with Streamlit\n📅 {datetime.now().strftime('%B %Y')}")

# ============================================
# MAIN CONTENT
# ============================================

# Hero Section
st.markdown('<div class="gradient-text">CIFAR-10 Vision AI</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">State-of-the-art Deep Learning for Image Classification</p>', unsafe_allow_html=True)

# Main columns
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-card">
        <h2 style="text-align: center; margin-bottom: 1rem;">📤 Upload Image</h2>
        <p style="text-align: center; color: #666;">Drag & drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🖼️ Your Image", use_container_width=True)
        
        # Image info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Width", f"{image.size[0]}px")
        with col_info2:
            st.metric("Height", f"{image.size[1]}px")
        with col_info3:
            st.metric("Mode", image.mode)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <h2 style="text-align: center; margin-bottom: 1rem;">🎯 Classification Result</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("🧠 Analyzing image with AI..."):
            img_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                all_probs = probs[0].cpu().numpy()
        
        # Animated result box
        st.markdown(f"""
        <div class="result-box">
            <div class="result-text">{CLASSES[pred]}</div>
            <div class="confidence-text">Confidence: {conf*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        st.markdown("#### Confidence Gauge")
        st.progress(conf)
        
        # Metrics row
        st.markdown("#### Performance Metrics")
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{conf*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_name.split('_')[0]}</div>
                <div class="metric-label">Architecture</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_acc:.1f}%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability distribution chart
        st.markdown("#### 📊 Probability Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71' if i == pred else '#95a5a6' for i in range(10)]
        bars = ax.barh(CLASSES, all_probs * 100, color=colors, edgecolor='white', linewidth=2)
        ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar, prob in zip(bars, all_probs):
            if prob * 100 > 5:
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{prob*100:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.9); border-radius: 20px;">
            <h3 style="color: #667eea;">✨ Ready for Classification</h3>
            <p style="color: #666;">Upload an image from the left panel to see the AI in action!</p>
            <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">Supports: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FEATURES SECTION
# ============================================

st.markdown("---")
st.markdown("### 🌟 Key Features")

feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)

with feat_col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem;">🧠</div>
        <div class="metric-value">Deep Learning</div>
        <div class="metric-label">ResNet-style CNN architecture</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem;">⚡</div>
        <div class="metric-value">Real-time</div>
        <div class="metric-label">Fast inference & response</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem;">🎯</div>
        <div class="metric-value">90%+ Accuracy</div>
        <div class="metric-label">State-of-the-art performance</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem;">🖼️</div>
        <div class="metric-value">10 Classes</div>
        <div class="metric-label">Complete CIFAR-10 coverage</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
    <p>Built with ❤️ using PyTorch & Streamlit | Trained on CIFAR-10 Dataset</p>
    <p style="font-size: 0.8rem;">© 2024 CIFAR-10 Vision AI - Advanced Image Classification System</p>
</div>
""", unsafe_allow_html=True)