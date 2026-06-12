import streamlit as st
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
from utils import scan_dataset
import time
import math
import numpy as np
import torch.nn as nn
from pathlib import Path as _Path
import os
import json
import asyncio
import tempfile

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
except Exception:
    genai = None

try:
    import edge_tts
except Exception:
    edge_tts = None

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="AgroVision AI | Crop Disease Classifier", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- INFERENCE HELPERS ---
def tta_images(pil_img):
    """Return a small set of augmented PIL images for Test-Time Augmentation."""
    imgs = [pil_img]
    try:
        imgs.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
        imgs.append(pil_img.rotate(10, resample=Image.BILINEAR))
        imgs.append(pil_img.rotate(-10, resample=Image.BILINEAR))
    except Exception:
        pass
    return imgs

def pil_to_tensor(img):
    """Convert PIL image to normalized tensor matching training preprocessing."""
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img)

def load_temperature(path='temperature.txt'):
    p = _Path(path)
    if p.exists():
        try:
            return float(p.read_text().strip())
        except Exception:
            return 1.0
    return 1.0

def _clean_json(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except Exception:
                return None
    return None

@st.cache_resource
def get_gemini_client():
    if load_dotenv is not None:
        load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "GEMINI_API_KEY is missing in your .env file"
    if genai is None:
        return None, "google-genai package is not installed"
    try:
        return genai.Client(api_key=api_key), ""
    except Exception as exc:
        return None, f"Gemini client init failed: {exc}"

@st.cache_data(ttl=86400, show_spinner=False)
def generate_monolingual_advice(predicted_label, confidence, target_lang):
    """Generates expert treatment data specifically mapped to one selected language key."""
    client, err = get_gemini_client()
    
    # Fallback dictionaries if API goes offline or credentials fail
    fallback_data = {
        "english": {
            "cause": f"Disease identified: {predicted_label}.",
            "situation": "Monitor affected plants closely for progression.",
            "solution": "Apply recommended fungicides or pesticides as per local guidelines.",
            "prevention": "Maintain crop rotation and proper irrigation practices."
        },
        "hindi": {
            "cause": f"पहचाना गया रोग: {predicted_label}।",
            "situation": "प्रभावित पौधों की नियमित निगरानी करें।",
            "solution": "स्थानीय दिशानिर्देशों के अनुसार उपयुक्त कवकनाशी या कीटनाशक लागू करें।",
            "prevention": "फसल चक्र और सही सिंचाई प्रथाओं का पालन करें।"
        },
        "haryanvi": {
            "cause": f"पहचानो गो रोग: {predicted_label}।",
            "situation": "प्रभावित बूटों की नित निगरानी करो।",
            "solution": "स्थानीय सलाह मुताबिक सही दवा या कीटनाशक लगाओ।",
            "prevention": "फसल का बदलाव और सही सिंचाई का रीवाज रखो।"
        }
    }
    
    if client is None:
        res = fallback_data.get(target_lang)
        res["_error"] = err
        return res

    # Tailor requirements instruction block by target option
    lang_requirements = {
        "english": "Provide the details clearly in pure English text context.",
        "hindi": "Provide text strictly written in Devanagari Script (Standard Hindi Form).",
        "haryanvi": "Provide text strictly in localized Haryanvi regional language dialect using Devanagari script."
    }

    prompt = f"""
You are an expert agriculture disease consultant for Indian farmers. Provide comprehensive, practical guidance.

Predicted disease/class: {predicted_label}
Model confidence: {confidence:.2f}%
Target Requested Language Output: {target_lang.upper()}

Return ONLY valid JSON with this exact schema (no extra markdown blocks, code blocks or exterior commentary text):
{{
  "cause": "...",
  "situation": "...",
  "solution": "...",
  "prevention": "..."
}}

Detailed Content Instructions:
1. "cause": Explain what causes this disease. 4-5 sentences. {lang_requirements.get(target_lang, "")}
2. "situation": Describe the current impact, spread risk. 4-5 sentences. {lang_requirements.get(target_lang, "")}
3. "solution": List specific treatments, dosages. Be very practical. 4-5 sentences. {lang_requirements.get(target_lang, "")}
4. "prevention": Provide preventive measures, resistant varieties. 4-5 sentences. {lang_requirements.get(target_lang, "")}
"""
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        parsed = _clean_json(getattr(response, "text", ""))
        if parsed and all(k in parsed for k in ["cause", "situation", "solution", "prevention"]):
            return parsed
    except Exception as exc:
        res = fallback_data.get(target_lang)
        res["_error"] = f"Gemini call failed: {exc}"
        return res
        
    res = fallback_data.get(target_lang)
    res["_error"] = "Gemini returned invalid structural matrix JSON"
    return res

async def _tts_bytes_async(text, voice):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        out_path = fp.name
    try:
        communicator = edge_tts.Communicate(text=text, voice=voice)
        await communicator.save(out_path)
        return Path(out_path).read_bytes()
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass

@st.cache_data(ttl=86400, show_spinner=False)
def text_to_speech_bytes(text, voice):
    if edge_tts is None or not text.strip():
        return None
    try:
        return asyncio.run(_tts_bytes_async(text, voice))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_tts_bytes_async(text, voice))
        finally:
            loop.close()

# --- HIGH CONTRACT / FUTURISTIC NEON GLASSMORPHISM CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 80% 20%, rgba(0, 229, 255, 0.05), transparent 50%),
                    radial-gradient(circle at 20% 80%, rgba(218, 0, 255, 0.04), transparent 50%),
                    #070d19 !important;
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        color: #f0f4f8 !important;
    }
    
    [data-testid="stHeader"], stDecoration, .stException, [data-test="stNotification"] {
        display: none !important;
    }
    div[data-testid="stImageHoverContainer"] + div {
        display: none !important;
    }
    
    [data-testid="stAppViewContainer"] > main .block-container {
        max-width: 1450px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    .brand-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .brand-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #00E5FF 0%, #90CAF9 50%, #DA00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-version {
        background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(218,0,255,0.1));
        border: 1px solid rgba(0, 229, 255, 0.3);
        color: #00E5FF;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 13px;
    }
    
    .glass-card {
        background: rgba(13, 25, 48, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .dataset-badge-container {
        display: flex;
        gap: 16px;
    }
    .dataset-mini-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00E5FF;
        padding: 10px 16px;
        border-radius: 0 12px 12px 0;
        flex: 1;
    }
    
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] [data-baseweb="select"] {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 2px dashed rgba(0, 229, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #00B0FF 0%, #00E5FF 100%) !important;
        color: #050a14 !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 229, 255, 0.4) !important;
        filter: brightness(1.1);
    }
    
    .prediction-panel {
        background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(218,0,255,0.05));
        border: 1px solid rgba(0, 229, 255, 0.2);
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        margin-top: 10px;
    }
    .badge-alert {
        background: #ff1744 !important;
        color: white !important;
    }
    .badge-success {
        background: #00E676 !important;
        color: #050a14 !important;
    }
    
    .class-pill-box {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
        color: #ffffff;
        border: 1px solid rgba(255,255,212,0.1);
    }
    
    .info-block {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-left: 3px solid #DA00FF;
    }
    .info-title {
        font-size: 14px;
        color: #90CAF9;
        font-weight: 700;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- BRANDING LAYER ---
st.markdown(
    """
    <div class="brand-container">
        <div>
            <div class="brand-title">AGROVISION AI</div>
            <div style="color: #b0bec5; font-size: 14px; margin-top: 4px;">Neural Network Crop Health Diagnostics & Multilingual Management System</div>
        </div>
        <div class="brand-version">v2.0 (Stable)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SCAN DATASET INITIALIZATION ---
root = Path('Train')
if not root.exists():
    classes, counts, total = [], {}, 0
else:
    classes, counts, total = scan_dataset(root)

# --- MAIN RESPONSIVE DASHBOARD LAYOUT ---
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    # Card 1: System Control & Dataset Overview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Dataset Infrastructure</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dataset-badge-container">
            <div class="dataset-mini-card">
                <div style="font-size: 12px; color: #b0bec5;">DIAGNOSTIC CLASSES</div>
                <div style="font-size: 24px; font-weight: 800; color: #ffffff;">{len(classes)} Target Classes</div>
            </div>
            <div class="dataset-mini-card" style="border-left-color: #DA00FF;">
                <div style="font-size: 12px; color: #b0bec5;">TOTAL IMAGE BANK</div>
                <div style="font-size: 24px; font-weight: 800; color: #ffffff;">{total} Imgs</div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 2: Interactive Ingestion Deck
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Crop Image Ingestion & Controls</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Image Block", type=['jpg', 'jpeg', 'png'], key="main_input", label_visibility="collapsed")
    
    st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
    model_path = st.text_input('Runtime Engine Weights Checkpoint (.pth)', 'model.pth')
    
    # NEW LANGUAGE INTERFACE SELECTOR DROPDOWN
    st.markdown('<div style="font-size:13px; font-weight:600; color:#90CAF9; margin-bottom: 2px;">Select Target Diagnosis Language</div>', unsafe_allow_html=True)
    selected_lang_label = st.selectbox(
        "Language Selection Deck",
        ["🇬🇧 English Framework", "🇮🇳 हिन्दी परामर्श (Hindi)", "🚜 हरियाणवी ब्योरा (Haryanvi)"],
        label_visibility="collapsed"
    )
    
    # Map selection back to runtime system data identifier keys
    lang_mapping = {
        "🇬🇧 English Framework": "english",
        "🇮🇳 हिन्दी परामर्श (Hindi)": "hindi",
        "🚜 हरियाणवी ब्योरा (Haryanvi)": "haryanvi"
    }
    chosen_lang = lang_mapping[selected_lang_label]
    
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    predict_btn = st.button('🚀 INITIALIZE COGNITIVE DIAGNOSIS')
    st.markdown('</div>', unsafe_allow_html=True)

    # Core Execution & Prediction block
    if predict_btn:
        if not uploaded:
            st.error("⚠️ Ingestion Error: Please feed an image execution matrix first.")
        elif not Path(model_path).exists():
            st.error(f"📂 Weights Error: Checkpoint path initialization failed for target: {model_path}")
        else:
            img = Image.open(uploaded).convert('RGB')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            with st.spinner('⚡ Mounting Deep Neural Weights Matrix...'):
                ck = torch.load(model_path, map_location=device)
                model_classes = ck.get('classes', ck.get('class_names', classes))
                model = models.resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, len(model_classes))
                model.load_state_dict(ck['model_state'])
                model.to(device)
                model.eval()

                # TTA Scaling Matrix
                temperature = load_temperature()
                tta_imgs = tta_images(img)
                probs_sum = None
                
                with torch.no_grad():
                    for pil in tta_imgs:
                        x = pil_to_tensor(pil).unsqueeze(0).to(device)
                        logits = model(x)
                        if temperature != 1.0:
                            logits = logits / (temperature if temperature > 0 else 1.0)
                        p = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                        probs_sum = p if probs_sum is None else probs_sum + p

                probs_avg = probs_sum / len(tta_imgs)
                top_idx = int(np.argmax(probs_avg))
                predicted_label = str(model_classes[top_idx])
                pct = float(probs_avg[top_idx] * 100.0)

            # Display Prediction Badge Container
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 AI Inference Analysis</div>', unsafe_allow_html=True)
            
            status_badge_class = "badge-success" if pct >= 50.0 else "badge-alert"
            st.markdown(
                f"""
                <div class="prediction-panel">
                    <div style="font-size: 14px; color:#b0bec5; margin-bottom:6px; font-weight:600; letter-spacing:1px;">IDENTIFIED PHENOTYPE</div>
                    <div style="font-size: 22px; font-weight:800; color:#fff; margin-bottom:8px;">{predicted_label}</div>
                    <span style="padding: 6px 16px; border-radius:20px; font-size:13px; font-weight:800; display:inline-block;" class="{status_badge_class}">
                        CONFIDENCE RATE: {pct:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Language Model Consultation Extraction Block (Monolingual optimization)
            with st.spinner(f'🤖 Querying Gemini Agribusiness Agent Model ({chosen_lang.upper()})...'):
                advice = generate_monolingual_advice(predicted_label, pct, chosen_lang)

            if advice.get('_error'):
                st.sidebar.error(f"LLM Network Warning: {advice['_error']}")

            # Multi-lingual Tab Layout Deployment (Filters out text to show ONLY selected language blocks)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">🌿 Agronomist Treatment Protocols ({selected_lang_label})</div>', unsafe_allow_html=True)
            
            if chosen_lang == "english":
                st.markdown(f'<div class="info-block"><div class="info-title">🔍 Etiology / Primary Cause</div>{advice.get("cause", "Data stream offline")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block"><div class="info-title">📍 Pathological Spread Situation</div>{advice.get("situation", "Data stream offline")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E5FF"><div class="info-title">💊 Chemical & Biological Prescription Remediation</div>{advice.get("solution", "Data stream offline")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E676"><div class="info-title">🛡️ Proactive Inoculation & Long-Term Prevention</div>{advice.get("prevention", "Data stream offline")}</div>', unsafe_allow_html=True)
                
            elif chosen_lang == "hindi":
                st.markdown(f'<div class="info-block"><div class="info-title">🔍 संक्रमण का मुख्य कारण</div>{advice.get("cause", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block"><div class="info-title">📍 वर्तमान प्रकोप की स्थिति</div>{advice.get("situation", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E5FF"><div class="info-title">💊 सटीक रासायनिक एवं जैविक उपचार प्रणाली</div>{advice.get("solution", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E676"><div class="info-title">🛡️ दीर्घकालिक फसल बचाव एवं रोकथाम नियंत्रण</div>{advice.get("prevention", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                
            elif chosen_lang == "haryanvi":
                st.markdown(f'<div class="info-block"><div class="info-title">🔍 बीमारी होण का असली कारण</div>{advice.get("cause", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block"><div class="info-title">📍 खेत में बीमारी के मौजूदा हालात</div>{advice.get("situation", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E5FF"><div class="info-title">💊 पक्का इलाज और दवाई की मात्र की जानकारी</div>{advice.get("solution", "अनुपलब्ध")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block" style="border-left-color:#00E676"><div class="info-title">🛡️ आगै बीमारी ना फैले उस खातर खास देसी व वैज्ञानिक तरीके</div>{advice.get("prevention", "अनुपलब्ध")}</div>', unsafe_allow_html=True)

            # Audio Speech Rendering Engine Layer (Generates ONLY the one selected speech track)
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 15px; font-weight:700; color:#00E5FF; margin-bottom:10px;">🔊 Neural TTS Voice Modulation Systems</div>', unsafe_allow_html=True)
            
            if edge_tts is None:
                st.warning("Voice modules are currently running offline.")
            else:
                # Compile matching dynamic audio text configuration string based on chosen language
                if chosen_lang == "english":
                    speech_text = f"Predicted disease matrix identifies {predicted_label}. Cause profile: {advice.get('cause', '')}. Suggested treatment regimen: {advice.get('solution', '')}"
                    voice_model = 'en-IN-PrabhatNeural'
                elif chosen_lang == "hindi":
                    speech_text = f"पहचाना गया रोग है {predicted_label}. मुख्य कारण: {advice.get('cause', '')}. उपचार विवरण: {advice.get('solution', '')}"
                    voice_model = 'hi-IN-SwaraNeural'
                else:  # haryanvi
                    speech_text = f"बीमारी पकड़ी गई है यो है {predicted_label}. यो होण का कारण: {advice.get('cause', '')}. इसका इलाज सुनो: {advice.get('solution', '')}"
                    voice_model = 'hi-IN-MadhurNeural'

                # Execute rendering engine for single voice tracking
                st.caption(f"📍 Active Channel: {selected_lang_label}")
                raw_audio = text_to_speech_bytes(speech_text, voice_model)
                if raw_audio:
                    st.audio(raw_audio, format='audio/mp3')
                else:
                    st.write("Voice synthesis processing failure")
                    
            st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # Card 3: Visual Real-time Analytics Feed Desk
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👁️ Live Inspection Viewfinder</div>', unsafe_allow_html=True)
    if uploaded:
        img_preview = Image.open(uploaded)
        st.image(img_preview, caption='Target matrix array loaded into runtime processing heap.', use_container_width=True)
    else:
        st.markdown(
            """
            <div style="border: 2px dashed rgba(255,255,255,0.05); border-radius:12px; height: 320px; display: flex; flex-direction:column; align-items: center; justify-content: center; color: #b0bec5;">
                <div style="font-size: 40px; margin-bottom: 10px;">🖼️</div>
                <div style="font-size:14px; font-weight:600;">System idle. Upload a crop leaf matrix asset for display framework output.</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 4: Sequenced Analytics Class Map Exploration Deck
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧬 Target Neural Output Class Distribution Map</div>', unsafe_allow_html=True)
    show_counts_btn = st.button('🔍 EXTRACT NEURAL TARGET MAP MATRIX', key='system_counts_trigger')
    
    if show_counts_btn:
        if not counts:
            st.info("Network dictionary maps are empty or not computed.")
        else:
            ordered_keys = sorted(counts.keys())
            badge_builder_accum = ""
            badges_placeholder = st.empty()
            
            for index, key in enumerate(ordered_keys):
                instances = counts[key]
                hue_spectrum = int((index * 360 / len(ordered_keys)) % 360)
                inline_style = f"background: linear-gradient(135deg, hsl({hue_spectrum}, 70%, 25%), hsl({(hue_spectrum+40)%360}, 60%, 15%)); border-color: hsl({hue_spectrum}, 80%, 50%);"
                
                badge_builder_accum += f'<span class="class-pill-box" style="{inline_style}">{key}: <b>{instances}</b></span>'
                badges_placeholder.markdown(f'<div style="max-height: 400px; overflow-y: auto;">{badge_builder_accum}</div>', unsafe_allow_html=True)
                time.sleep(0.04)
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom UX Enhancement Banner
st.markdown(
    """
    <div class="glass-card" style="background: linear-gradient(90deg, rgba(7,13,25,0.8) 0%, rgba(0,229,255,0.03) 100%) !important;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="background: rgba(0, 229, 255, 0.1); border: 1px solid #00E5FF; color: #00E5FF; padding: 10px 14px; border-radius: 12px; font-weight:800; font-size:14px;">PRO-TIP</div>
            <div style="font-size: 13.5px; color: #b0bec5; line-height: 1.5;">
                <b>Live Vision Mode active.</b> For optimal runtime accuracy, use high-definition macro target images captured in balanced light matrices. Avoid multi-leaf clusters or shaded validation frames.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)