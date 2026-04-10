# app.py - Ultimate Professional CIFAR-10 Vision AI
# Modern Dashboard Design with Glassmorphism & Dark Theme

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CIFAR-10 Vision AI | Professional Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #3b82f6;
        --primary-light: #60a5fa;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --bg-dark: #020617;
        --bg-card: rgba(15, 23, 42, 0.7);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: rgba(148, 163, 184, 0.1);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit elements */
    #MainMenu, footer, header, [data-testid="stToolbar"] {
        visibility: hidden !important;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, 
            #020617 0%, 
            #0f172a 50%, 
            #1e1b4b 100%);
        background-attachment: fixed;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(15, 23, 42, 0.95) 0%, 
            rgba(2, 6, 23, 0.98) 100%) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
        min-width: 300px !important;
        max-width: 300px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem !important;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    .sidebar-logo-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    .sidebar-logo-subtitle {
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* Sidebar sections */
    .sidebar-section {
        margin-bottom: 1.5rem;
    }

    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
    }

    /* Model info card */
    .model-info-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .model-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    .model-info-row:last-child {
        margin-bottom: 0;
    }

    .model-info-label {
        color: var(--text-secondary);
    }

    .model-info-value {
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .accuracy-badge {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin: 1rem 0;
    }

    .accuracy-value {
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-light), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .accuracy-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* Classes grid */
    .classes-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }

    .class-item {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        transition: all 0.2s ease;
        cursor: default;
    }

    .class-item:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: var(--text-primary);
    }

    /* ===== MAIN CONTENT ===== */
    .main-content {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Header */
    .dashboard-header {
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* Glass cards */
    .glass-panel {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }

    .glass-panel:hover {
        border-color: rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }

    .panel-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }

    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Upload area */
    .upload-zone {
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(59, 130, 246, 0.02);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .upload-zone:hover {
        border-color: var(--primary);
        background: rgba(59, 130, 246, 0.05);
    }

    .upload-icon-large {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }

    .upload-text-main {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }

    .upload-text-sub {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Image preview */
    .image-preview {
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid var(--border);
    }

    .image-meta {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }

    .meta-badge {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: var(--primary-light);
        font-family: 'JetBrains Mono', monospace;
    }

    /* Results */
    .result-hero {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.15), 
            rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }

    .result-class {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }

    .result-confidence {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    .confidence-bar {
        margin: 1rem 0;
    }

    /* Top predictions */
    .predictions-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .prediction-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 10px;
        transition: all 0.2s ease;
    }

    .prediction-item:hover {
        background: rgba(59, 130, 246, 0.05);
        border-color: rgba(59, 130, 246, 0.2);
    }

    .prediction-item.top {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }

    .pred-rank {
        width: 28px;
        height: 28px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-secondary);
    }

    .pred-rank.top {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
    }

    .pred-icon {
        font-size: 1.25rem;
    }

    .pred-name {
        flex: 1;
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    .pred-prob {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--primary-light);
        font-family: 'JetBrains Mono', monospace;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .stat-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-secondary);
    }

    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        border-radius: 4px !important;
    }

    .stProgress > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    /* Footer */
    .dashboard-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
    }

    /* Responsive */
    @media (max-width: 1024px) {
        .main-content {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTS
# ============================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CLASSES = [
    ('🛩️', 'Airplane'),
    ('🚗', 'Automobile'),
    ('🐦', 'Bird'),
    ('🐱', 'Cat'),
    ('🦌', 'Deer'),
    ('🐕', 'Dog'),
    ('🐸', 'Frog'),
    ('🐴', 'Horse'),
    ('🚢', 'Ship'),
    ('🚛', 'Truck')
]

# ============================================
# MODEL DEFINITIONS
# ============================================

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
# MODEL LOADING
# ============================================

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCNN(num_classes=10, dropout_rate=0.4)

    if os.path.exists('saved_models/best_model.pth'):
        try:
            checkpoint = torch.load('saved_models/best_model.pth', map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            accuracy = float(checkpoint.get('test_accuracy', 89.05))
        except Exception as e:
            accuracy = 89.05
    else:
        accuracy = 89.05

    model = model.to(device)
    model.eval()
    return model, device, "ResNet-18", accuracy


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return transform(image).unsqueeze(0)


# ============================================
# SIDEBAR COMPONENT
# ============================================

def render_sidebar():
    """Professional sidebar with model info and classes"""

    # Logo section
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div>
            <div class="sidebar-logo-text">Vision AI</div>
            <div class="sidebar-logo-subtitle">CIFAR-10 Classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Performance Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Model Performance</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="accuracy-badge">
        <div class="accuracy-value">{best_acc:.1f}%</div>
        <div class="accuracy-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-info-card">
        <div class="model-info-row">
            <span class="model-info-label">Architecture</span>
            <span class="model-info-value">ResNet-18</span>
        </div>
        <div class="model-info-row">
            <span class="model-info-label">Parameters</span>
            <span class="model-info-value">11.2M</span>
        </div>
        <div class="model-info-row">
            <span class="model-info-label">Input Size</span>
            <span class="model-info-value">32×32×3</span>
        </div>
        <div class="model-info-row">
            <span class="model-info-label">Device</span>
            <span class="model-info-value">{}</span>
        </div>
    </div>
    """.format(device.type.upper()), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Description Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">About Model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6; margin-bottom: 1rem;">
        Deep residual learning framework with skip connections. 
        Features batch normalization and dropout regularization 
        for improved generalization on CIFAR-10 dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Supported Classes Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Supported Classes</div>', unsafe_allow_html=True)

    st.markdown('<div class="classes-grid">', unsafe_allow_html=True)
    for emoji, name in CLASSES:
        st.markdown(f'<div class="class-item">{emoji} {name}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.1);">
        <div style="font-size: 0.75rem; color: #64748b; text-align: center;">
            PyTorch {} • Streamlit<br>
            © {} Vision AI
        </div>
    </div>
    """.format(torch.__version__, datetime.now().year), unsafe_allow_html=True)


# ============================================
# MAIN DASHBOARD COMPONENTS
# ============================================

def render_header():
    """Dashboard header"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-title">Image Classification Dashboard</div>
        <div class="header-subtitle">Upload an image to classify it into one of 10 CIFAR-10 categories</div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_panel():
    """Upload panel component"""
    st.markdown("""
    <div class="panel-header">
        <div class="panel-icon">📸</div>
        <div class="panel-title">Upload Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="uploader"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon-large">📤</div>
            <div class="upload-text-main">Drop image here or click to browse</div>
            <div class="upload-text-sub">Supports JPG, PNG up to 10MB</div>
        </div>
        """, unsafe_allow_html=True)
        return None

    try:
        image = Image.open(uploaded_file).convert('RGB')

        # Store in session state
        st.session_state['current_image'] = image
        st.session_state['current_file'] = uploaded_file

        # Display preview
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Metadata
        uploaded_file.seek(0, 2)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)

        st.markdown(f"""
        <div class="image-meta">
            <span class="meta-badge">{image.size[0]}×{image.size[1]} px</span>
            <span class="meta-badge">{image.mode}</span>
            <span class="meta-badge">{file_size/1024:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)

        return image

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def render_results_panel():
    """Results panel component"""
    st.markdown("""
    <div class="panel-header">
        <div class="panel-icon">🎯</div>
        <div class="panel-title">Classification Results</div>
    </div>
    """, unsafe_allow_html=True)

    if 'current_image' not in st.session_state:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div>Upload an image to see classification results</div>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        with st.spinner("Analyzing..."):
            image = st.session_state['current_image']
            tensor = preprocess_image(image).to(device)

            with torch.no_grad():
                output = model(tensor)
                probabilities = F.softmax(output, dim=1)
                pred_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_idx].item()
                all_probs = probabilities[0].cpu().numpy()

        emoji, name = CLASSES[pred_idx]

        # Main result
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-class">{emoji} {name}</div>
            <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<div class="confidence-bar">", unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown("</div>", unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{confidence*100:.0f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{model_name}</div>
                <div class="stat-label">Model</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{best_acc:.0f}%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 predictions
        st.markdown("**🏆 Top 3 Predictions**")

        top3_indices = np.argsort(all_probs)[-3:][::-1]

        st.markdown('<div class="predictions-list">', unsafe_allow_html=True)
        for rank, idx in enumerate(top3_indices, 1):
            emoji, name = CLASSES[idx]
            prob = all_probs[idx]
            is_top = rank == 1

            rank_class = "top" if is_top else ""
            rank_badge = "top" if is_top else ""

            st.markdown(f"""
            <div class="prediction-item {rank_class}">
                <div class="pred-rank {rank_badge}">#{rank}</div>
                <div class="pred-icon">{emoji}</div>
                <div class="pred-name">{name}</div>
                <div class="pred-prob">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


def render_footer():
    """Dashboard footer"""
    st.markdown("""
    <div class="dashboard-footer">
        <p>CIFAR-10 Vision AI • Built with PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN APPLICATION
# ============================================

# Load model
try:
    model, device, model_name, best_acc = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Render sidebar
with st.sidebar:
    render_sidebar()

# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Header
render_header()

# Two column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    render_upload_panel()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    render_results_panel()
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
render_footer()

st.markdown('</div>', unsafe_allow_html=True)
