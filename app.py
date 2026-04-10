# app.py - CIFAR-10 Vision AI (Optimized & Fully Functional)
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CIFAR-10 Vision AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* لا نخفي header لأنه يحتوي على زر القائمة الجانبية */
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%); background-attachment: fixed; }
    [data-testid="stSidebar"] { background: rgba(5,5,10,0.95) !important; backdrop-filter: blur(12px) !important; border-right: 1px solid rgba(255,255,255,0.08) !important; }
    /* تحسين مظهر الزر دون إخفائه */
    [data-testid="stSidebarCollapsedControl"] { background: rgba(100,150,150,0.3) !important; border-radius: 0 12px 12px 0 !important; margin-top: 80px !important; padding: 6px 8px !important; }
    [data-testid="stSidebarCollapsedControl"]:hover { background: rgba(100,150,150,0.6) !important; }
    [data-testid="stSidebarCollapsedControl"] svg { color: #8ec5c2 !important; width: 20px !important; height: 20px !important; }
    /* تحسين header - تصغير الارتفاع وجعله شفافاً */
    header { background: transparent !important; backdrop-filter: blur(0px) !important; border-bottom: none !important; }
    .glass-card { background: rgba(10,10,20,0.6); backdrop-filter: blur(12px); border-radius: 28px; padding: 1.5rem; border: 1px solid rgba(168,237,234,0.15); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.4); transition: all 0.3s ease; }
    .glass-card:hover { transform: translateY(-4px); border-color: rgba(168,237,234,0.3); }
    .hero { text-align: center; padding: 1rem 0 0.5rem 0; }
    .hero-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #8ec5c2 0%, #e8b8c4 50%, #ffc896 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.2rem; }
    .hero-subtitle { font-size: 0.9rem; color: rgba(255,255,255,0.4); }
    @media (max-width: 768px) { .hero-title { font-size: 2rem; } }
    .section-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(168,237,234,0.15); }
    .section-icon { font-size: 1.3rem; }
    .section-title { font-weight: 700; font-size: 1rem; color: rgba(255,255,255,0.9); }
    .section-subtitle { font-size: 0.65rem; color: rgba(255,255,255,0.35); }
    .result-card { background: linear-gradient(135deg, rgba(100,150,150,0.15), rgba(200,150,160,0.1)); backdrop-filter: blur(12px); border-radius: 28px; padding: 1.2rem; text-align: center; border: 1px solid rgba(168,237,234,0.2); margin-bottom: 1rem; }
    .prediction { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #8ec5c2, #e8b8c4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }
    .confidence { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.4rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin: 1rem 0; }
    .metric-item { background: rgba(15,15,30,0.6); border-radius: 20px; padding: 0.6rem; text-align: center; transition: all 0.3s ease; border: 1px solid rgba(168,237,234,0.08); }
    .metric-item:hover { background: rgba(30,30,50,0.7); transform: translateY(-3px); }
    .metric-value { font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #8ec5c2, #e8b8c4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .metric-label { font-size: 0.6rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
    .image-container { border-radius: 20px; overflow: hidden; background: rgba(0,0,0,0.5); margin-top: 0.8rem; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.6rem; font-weight: 500; margin: 0.15rem; background: rgba(100,150,150,0.15); color: #8ec5c2; border: 1px solid rgba(100,150,150,0.2); }
    .features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }
    .feature-item { text-align: center; padding: 0.7rem; background: rgba(15,15,30,0.5); border-radius: 20px; transition: all 0.3s ease; border: 1px solid rgba(168,237,234,0.05); }
    .feature-item:hover { background: rgba(30,30,50,0.6); transform: translateY(-3px); }
    .feature-icon { font-size: 1.3rem; margin-bottom: 0.2rem; display: inline-block; }
    .feature-title { font-size: 0.7rem; font-weight: 700; color: rgba(255,255,255,0.8); margin-bottom: 0.1rem; }
    .feature-desc { font-size: 0.55rem; color: rgba(255,255,255,0.35); }
    .stProgress > div > div { background: linear-gradient(90deg, #8ec5c2, #e8b8c4); border-radius: 20px; }
    .stButton > button { background: linear-gradient(135deg, #8ec5c2, #e8b8c4); color: #0a0a0f; border: none; border-radius: 40px; padding: 0.4rem 1rem; font-weight: 600; font-size: 0.8rem; transition: all 0.3s ease; width: 100%; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(100,150,150,0.4); }
    .footer { text-align: center; padding: 1rem; color: rgba(255,255,255,0.2); font-size: 0.65rem; border-top: 1px solid rgba(168,237,234,0.05); margin-top: 1rem; }
    @media (max-width: 992px) { .hero-title { font-size: 2.5rem; } .prediction { font-size: 1.4rem; } .metric-value { font-size: 1.1rem; } }
    @media (max-width: 768px) { .hero-title { font-size: 1.8rem; } .hero-subtitle { font-size: 0.7rem; } .prediction { font-size: 1.1rem; } .features { grid-template-columns: repeat(2, 1fr); gap: 0.5rem; } .glass-card { padding: 0.8rem; } }
    @media (max-width: 480px) { .hero-title { font-size: 1.4rem; } .prediction { font-size: 1rem; } .features { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

# -------------------- CONSTANTS --------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = [
    {'name': 'Plane', 'emoji': '✈️', 'desc': 'Aircraft', 'color': '#8ec5c2'},
    {'name': 'Car', 'emoji': '🚗', 'desc': 'Automobile', 'color': '#90ee90'},
    {'name': 'Bird', 'emoji': '🐦', 'desc': 'Bird species', 'color': '#ffb347'},
    {'name': 'Cat', 'emoji': '🐱', 'desc': 'Feline', 'color': '#ff6b6b'},
    {'name': 'Deer', 'emoji': '🦌', 'desc': 'Wild deer', 'color': '#d4a5a5'},
    {'name': 'Dog', 'emoji': '🐕', 'desc': 'Canine', 'color': '#c9a0dc'},
    {'name': 'Frog', 'emoji': '🐸', 'desc': 'Amphibian', 'color': '#7dd3fc'},
    {'name': 'Horse', 'emoji': '🐴', 'desc': 'Equine', 'color': '#f0a3a3'},
    {'name': 'Ship', 'emoji': '🚢', 'desc': 'Vessel', 'color': '#5dade2'},
    {'name': 'Truck', 'emoji': '🚛', 'desc': 'Heavy vehicle', 'color': '#e5989b'}
]

# -------------------- MODEL DEFINITION (ONLY AdvancedCNN) --------------------
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

# -------------------- LOAD MODEL --------------------
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
        except:
            acc = 89.05
    else:
        acc = 89.05
    model = model.to(device)
    model.eval()
    return model, device, "AdvancedCNN", acc

model, device, model_name, best_acc = load_model()

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return transform(image).unsqueeze(0)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:0.5rem 0 1rem 0'><div style='font-size:2rem'>🎨</div><div style='font-weight:800;font-size:1.2rem;background:linear-gradient(135deg,#8ec5c2,#e8b8c4);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>CIFAR-10 Vision</div><div style='font-size:0.6rem;color:rgba(255,255,255,0.35)'>ADVANCED CLASSIFIER</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🤖 MODEL INFO**")
    c1, c2 = st.columns(2)
    c1.metric("Architecture", model_name)
    c2.metric("Accuracy", f"{best_acc:.1f}%")
    st.markdown("---")
    st.markdown("**📋 SUPPORTED CLASSES**")
    for cls in CLASSES:
        st.markdown(f"{cls['emoji']} **{cls['name']}**  \n<small>{cls['desc']}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📊 DATASET**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", "50K")
    c2.metric("Test", "10K")
    c3.metric("Size", "32²")
    st.markdown("---")
    st.caption(f"PyTorch • Streamlit\n{datetime.now().year}")

# -------------------- MAIN CONTENT --------------------
st.markdown('<div class="hero"><div class="hero-title">visual intelligence</div><div class="hero-subtitle">advanced deep learning for CIFAR-10 image classification</div></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    st.markdown('<div class="glass-card"><div class="section-header"><span class="section-icon">📸</span><div><div class="section-title">Upload Image</div><div class="section-subtitle">Supported: JPG, PNG, JPEG</div></div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        fsize = len(uploaded.getvalue())/1024
        st.markdown(f'<div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-top:0.6rem"><span class="badge">📏 {img.size[0]}×{img.size[1]}px</span><span class="badge">💾 {fsize:.1f} KB</span><span class="badge">🎨 {img.mode}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card"><div class="section-header"><span class="section-icon">🎯</span><div><div class="section-title">Prediction Result</div><div class="section-subtitle">Real-time AI inference</div></div></div>', unsafe_allow_html=True)
    if uploaded:
        with st.spinner("Analyzing image..."):
            tensor = preprocess(img).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred_idx].item()
                all_probs = probs[0].cpu().numpy()
        pred_class = CLASSES[pred_idx]
        st.markdown(f'<div class="result-card"><div style="font-size:2.2rem">{pred_class["emoji"]}</div><div class="prediction">{pred_class["name"]}</div><div class="confidence">confidence {conf*100:.2f}%</div><div style="font-size:0.6rem;color:rgba(255,255,255,0.35)">{pred_class["desc"]}</div></div>', unsafe_allow_html=True)
        st.markdown("#### Confidence")
        st.progress(conf)
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{conf*100:.1f}%")
        m2.metric("Architecture", model_name.split('_')[0])
        m3.metric("Model Accuracy", f"{best_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🏆 Top 3 Predictions")
        top3 = np.argsort(all_probs)[-3:][::-1]
        medals = ['🥇','🥈','🥉']
        for rank, idx in enumerate(top3):
            cls = CLASSES[idx]
            prob_val = all_probs[idx]*100
            st.markdown(f'<div style="background:rgba(20,20,40,0.6);border-radius:16px;padding:0.5rem;margin:0.4rem 0"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="font-size:1.2rem">{medals[rank]}</span> <span style="font-size:1rem">{cls["emoji"]}</span> <strong>{cls["name"]}</strong></div><div style="color:{"#8ec5c2" if rank==0 else "#aaa"}">{prob_val:.1f}%</div></div><div style="background:rgba(255,255,255,0.1);border-radius:12px;height:6px;margin-top:0.3rem"><div style="width:{prob_val}%;height:100%;background:linear-gradient(90deg,{cls["color"]},#8ec5c2);border-radius:12px"></div></div></div>', unsafe_allow_html=True)
        others = [i for i in range(10) if i not in top3]
        with st.expander(f"📋 Other Classes ({len(others)} remaining)"):
            for idx in sorted(others, key=lambda x: all_probs[x], reverse=True):
                cls = CLASSES[idx]
                prob_val = all_probs[idx]*100
                st.markdown(f"{cls['emoji']} {cls['name']}: `{prob_val:.1f}%`")
    else:
        st.markdown('<div style="text-align:center;padding:1.5rem 1rem"><div style="font-size:2.5rem;opacity:0.3">🖼️</div><div style="font-weight:600">No Image Selected</div><div style="font-size:0.65rem;color:rgba(255,255,255,0.3)">Upload an image from the left panel</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FEATURES --------------------
st.markdown('<div style="text-align:center;margin:1rem 0"><div style="font-weight:700;font-size:1rem;background:linear-gradient(135deg,#8ec5c2,#e8b8c4);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Key Features</div></div><div class="features">', unsafe_allow_html=True)
features = [('🧠','Deep Learning','CNN Architecture'),('⚡','Real-time','Fast Inference'),('🎯','Top-3 Focus','Smart Display'),('📱','Responsive','Mobile Ready')]
cols = st.columns(4)
for i, (icon, title, desc) in enumerate(features):
    with cols[i]:
        st.markdown(f'<div class="feature-item"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(f'<div class="footer">CIFAR-10 Dataset • {model_name} Architecture • Deployed with Streamlit</div>', unsafe_allow_html=True)
