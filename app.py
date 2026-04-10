# app.py - واجهة احترافية فاخرة مع تحسينات الأداء

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="CIFAR-10 Vision AI | Advanced Classifier",
    page_icon="🎯",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    # Import model architecture (assuming it's defined in the same file or imported)
    # For brevity, we're loading from saved files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('saved_models/results_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Load the best model (simplified - in production, import the actual model class)
    checkpoint = torch.load('saved_models/best_model.pth', map_location=device)
    
    # Placeholder for actual model loading
    # model = get_model_class(checkpoint['model_name'])()
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint, device, summary

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    return transform(image).unsqueeze(0)

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-title { text-align: center; font-size: 3rem; font-weight: bold; color: white; margin-bottom: 2rem; }
    .result-card { background: white; border-radius: 20px; padding: 2rem; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
    .prediction-text { font-size: 2rem; font-weight: bold; color: #667eea; }
    .confidence-text { font-size: 1.2rem; color: #666; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-title">🎯 CIFAR-10 Vision AI</div>', unsafe_allow_html=True)

# Load model once
checkpoint, device, summary = load_model()
classes = checkpoint.get('classes', summary.get('classes', 
    ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')))

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Your Image", use_container_width=True)

with col2:
    st.markdown("### 🎯 Classification Result")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            input_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                # Placeholder for actual inference
                # output = model(input_tensor)
                # probs = torch.softmax(output, dim=1)
                # pred = torch.argmax(probs, dim=1).item()
                # confidence = probs[0][pred].item()
                
                # Mock result (replace with actual)
                pred, confidence = 0, 0.95
            
            st.markdown(f"""
            <div class="result-card">
                <div class="prediction-text">{classes[pred]} ✨</div>
                <div class="confidence-text">Confidence: {confidence*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(confidence)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Architecture", summary.get('best_model', 'CNN').split('_')[0])
            with col_b:
                st.metric("Accuracy", f"{summary.get('best_accuracy', 89):.1f}%")
            with col_c:
                st.metric("Classes", "10")
    else:
        st.info("👈 Upload an image to start classification")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: rgba(255,255,255,0.7);'>Powered by PyTorch | Deployed with Streamlit | {datetime.now().strftime('%B %Y')}</div>", unsafe_allow_html=True)
