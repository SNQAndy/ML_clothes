import streamlit as st
import pandas as pd
import random
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import json
import torch
import cv2
from collections import Counter
import torchvision.transforms as transforms

#test for Streamlitdeploy:
import sys, os
sys.path.append(os.path.dirname(__file__))


# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Fashion Swipe & Generate",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Importiere das Pix2Pix Turbo Modell
try:
    from pix2pix_turbo import Pix2PixTurbo
    PIX2PIX_AVAILABLE = True
except ImportError:
    PIX2PIX_AVAILABLE = False
    st.warning("Pix2Pix Turbo Modell nicht verfügbar. Verwende Mock-Generierung.")



# Custom CSS (gleich wie vorher)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B9D;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .fashion-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
        transition: transform 0.3s ease-out, opacity 0.3s ease-out;
        position: relative;
        cursor: grab;
        user-select: none;
        touch-action: none;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    
    .generate-container {
        background: linear-gradient(135deg, #FF6B9D 0%, #4ECDC4 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 30px 0;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .generated-result {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 30px 0;
        text-align: center;
    }
    
    .progress-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Fashion-MNIST Klassen
FASHION_CLASSES = {
    0: "T-Shirt/Top",
    1: "Hose", 
    2: "Pullover",
    3: "Kleid",
    4: "Mantel",
    5: "Sandalen",
    6: "Hemd",
    7: "Sneaker",
    8: "Tasche",
    9: "Stiefeletten"
}

# Pix2Pix Modell laden
@st.cache_resource
def load_pix2pix_model():
    """Lädt das Pix2Pix Turbo Modell"""
    if not PIX2PIX_AVAILABLE:
        return None
    
    try:
        # Lade das vortrainierte Modell für sketch_to_image
        model = Pix2PixTurbo(pretrained_name="sketch_to_image_stochastic")
        model.set_eval()
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Pix2Pix Modells: {e}")
        return None

# Fashion-MNIST Daten laden
@st.cache_data
def load_fashion_mnist():
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_all = np.concatenate([x_train, x_test])
        y_all = np.concatenate([y_train, y_test])
        return x_all, y_all
    except Exception as e:
        st.error(f"Fehler beim Laden von Fashion-MNIST: {e}")
        return None, None

def create_realistic_fashion_sketch(selected_items, canvas_size=(512, 512)):
    """Erstellt eine realistische Fashion-Skizze basierend auf den ausgewählten Items"""
    
    # Erstelle leere Leinwand
    sketch = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(sketch)
    
    # Analysiere die Kategorien
    categories = [item['category'] for item in selected_items]
    category_counts = Counter(categories)
    
    # Zeichne basierend auf den häufigsten Kategorien
    y_center = canvas_size[1] // 2
    x_center = canvas_size[0] // 2
    
    # Körper-Grundform
    draw.ellipse([x_center-15, 80, x_center+15, 120], outline='black', width=2)  # Kopf
    draw.line([x_center, 120, x_center, 350], fill='black', width=3)  # Torso
    
    for category, count in category_counts.most_common():
        line_width = max(2, min(5, count))  # Dickere Linien für häufigere Kategorien
        
        if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
            # Oberteil mit realistischeren Proportionen
            draw.rectangle([x_center-60, 120, x_center+60, 200], outline='black', width=line_width)
            # Ärmel
            draw.line([x_center-60, 140, x_center-90, 180], fill='black', width=line_width)
            draw.line([x_center+60, 140, x_center+90, 180], fill='black', width=line_width)
            # Details
            draw.line([x_center-40, 130, x_center+40, 130], fill='black', width=1)  # Kragen
            
        elif category == "Kleid":
            # Kleid-Silhouette
            points = [
                (x_center-50, 120), (x_center+50, 120),  # Schultern
                (x_center+70, 300), (x_center-70, 300)   # Saum
            ]
            draw.polygon(points, outline='black', width=line_width)
            # Ärmel
            draw.line([x_center-50, 140, x_center-80, 180], fill='black', width=line_width)
            draw.line([x_center+50, 140, x_center+80, 180], fill='black', width=line_width)
            
        elif category == "Hose":
            # Hose mit realistischen Beinen
            draw.rectangle([x_center-50, 200, x_center+50, 250], outline='black', width=line_width)  # Hüfte
            # Linkes Bein
            draw.line([x_center-25, 250, x_center-25, 400], fill='black', width=line_width)
            draw.line([x_center-45, 250, x_center-45, 400], fill='black', width=line_width)
            # Rechtes Bein
            draw.line([x_center+25, 250, x_center+25, 400], fill='black', width=line_width)
            draw.line([x_center+45, 250, x_center+45, 400], fill='black', width=line_width)
            
        elif category == "Mantel":
            # Mantel (länger als normales Oberteil)
            draw.rectangle([x_center-80, 120, x_center+80, 280], outline='black', width=line_width)
            # Lange Ärmel
            draw.line([x_center-80, 140, x_center-110, 220], fill='black', width=line_width)
            draw.line([x_center+80, 140, x_center+110, 220], fill='black', width=line_width)
            # Knöpfe
            for i in range(3):
                y_button = 140 + i * 40
                draw.ellipse([x_center-5, y_button, x_center+5, y_button+10], outline='black', width=1)
                
        elif category in ["Sneaker", "Stiefeletten", "Sandalen"]:
            # Schuhe
            draw.ellipse([x_center-45, 400, x_center-15, 430], outline='black', width=line_width)
            draw.ellipse([x_center+15, 400, x_center+45, 430], outline='black', width=line_width)
            if category == "Stiefeletten":
                # Höhere Schuhe
                draw.rectangle([x_center-45, 380, x_center-15, 400], outline='black', width=line_width)
                draw.rectangle([x_center+15, 380, x_center+45, 400], outline='black', width=line_width)
                
        elif category == "Tasche":
            # Handtasche
            draw.rectangle([x_center+60, 180, x_center+100, 220], outline='black', width=line_width)
            draw.line([x_center+70, 170, x_center+90, 170], fill='black', width=line_width)  # Griff
            draw.line([x_center+70, 170, x_center+70, 180], fill='black', width=line_width)
            draw.line([x_center+90, 170, x_center+90, 180], fill='black', width=line_width)
    
    # Konvertiere zu numpy array und erstelle Edge-Map
    sketch_array = np.array(sketch)
    
    # Edge Detection für klarere Linien
    gray = cv2.cvtColor(sketch_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Konvertiere zurück zu RGB
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Invertiere (weiße Linien auf schwarzem Hintergrund)
    edge_image = 255 - edge_image
    
    return edge_image

def generate_fashion_with_pix2pix(selected_items, style_prompt, model):
    """Generiert Fashion-Bilder mit dem echten Pix2Pix Turbo Modell"""
    
    if model is None:
        return generate_fashion_image_mock(selected_items, style_prompt)
    
    try:
        # Erstelle realistische Skizze
        sketch = create_realistic_fashion_sketch(selected_items)
        
        # Konvertiere zu PIL Image
        sketch_pil = Image.fromarray(sketch.astype(np.uint8))
        
        # Transformationen für das Modell
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Bereite Input vor
        input_tensor = transform(sketch_pil).unsqueeze(0).cuda()
        
        # Style-Prompt basierend auf Kategorien erstellen
        categories = [item['category'] for item in selected_items]
        category_text = ", ".join(set(categories))
        enhanced_prompt = f"High quality fashion photograph of {category_text}, {style_prompt}, professional lighting, clean background"
        
        # Generiere Bild mit dem Modell
        with torch.no_grad():
            generated_tensor = model(input_tensor, prompt=enhanced_prompt, deterministic=True)
        
        # Konvertiere zurück zu numpy
        generated_image = generated_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        generated_image = (generated_image + 1) / 2  # Denormalisierung [-1,1] -> [0,1]
        generated_image = np.clip(generated_image, 0, 1)
        
        return generated_image
        
    except Exception as e:
        st.error(f"Fehler bei der Bildgenerierung: {e}")
        # Fallback zur Mock-Generierung
        return generate_fashion_image_mock(selected_items, style_prompt)

def generate_fashion_image_mock(selected_items, style_prompt):
    """Verbesserte Mock-Funktion für realistischere Fashion-Bilder"""
    
    # Erstelle eine realistischere Skizze
    sketch = create_realistic_fashion_sketch(selected_items)
    
    # Erstelle ein realistischeres Fashion-Bild
    generated = np.ones((512, 512, 3), dtype=np.float32)
    
    # Analyse der Kategorien für intelligentere Farbwahl
    categories = [item['category'] for item in selected_items]
    category_counts = Counter(categories)
    
    # Verbesserte Farbpalette
    realistic_colors = {
        "T-Shirt/Top": [0.9, 0.9, 0.95],      # Helles Blau/Weiß
        "Hose": [0.2, 0.3, 0.6],              # Denim Blau
        "Pullover": [0.7, 0.4, 0.4],          # Warmes Rot
        "Kleid": [0.8, 0.2, 0.4],             # Elegantes Rosa
        "Mantel": [0.3, 0.3, 0.3],            # Dunkles Grau
        "Sneaker": [1.0, 1.0, 1.0],           # Weiß
        "Hemd": [0.95, 0.95, 1.0],            # Hellblau
        "Tasche": [0.4, 0.2, 0.1],            # Braun
        "Stiefeletten": [0.1, 0.1, 0.1],      # Schwarz
        "Sandalen": [0.8, 0.6, 0.4]           # Beige
    }
    
    # Erstelle Texturen und Muster
    for i in range(512):
        for j in range(512):
            # Gradient-Basis
            generated[i, j] = [0.98 - i/2048, 0.98 - i/2048, 0.98 - i/2048]
    
    # Füge realistische Kleidungsbereiche hinzu
    for idx, (category, count) in enumerate(category_counts.most_common()):
        if category in realistic_colors:
            color = realistic_colors[category]
            
            # Positioniere Kleidungsstücke realistisch
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                # Oberteil-Bereich
                generated[120:200, 190:322] = color
                # Füge Textur hinzu
                for i in range(120, 200, 4):
                    for j in range(190, 322, 4):
                        if (i + j) % 8 == 0:
                            generated[i:i+2, j:j+2] = [min(1.0, c + 0.1) for c in color]
                            
            elif category == "Kleid":
                # Kleid-Bereich
                generated[120:300, 175:337] = color
                
            elif category == "Hose":
                # Hosen-Bereich (untere Hälfte)
                generated[200:400, 180:332] = color
                # Jeans-Textur
                for i in range(200, 400, 8):
                    generated[i:i+2, 180:332] = [max(0.0, c - 0.1) for c in color]
                    
            elif category == "Mantel":
                # Mantel-Bereich (größer)
                generated[110:280, 160:352] = color
                
            elif category in ["Sneaker", "Stiefeletten", "Sandalen"]:
                # Schuh-Bereiche
                generated[400:430, 170:220] = color  # Linker Schuh
                generated[400:430, 292:342] = color  # Rechter Schuh
    
    # Kombiniere mit Sketch für mehr Details
    sketch_normalized = sketch.astype(np.float32) / 255.0
    
    # Intelligente Kombination: Verwende Sketch für Konturen
    sketch_mask = np.mean(sketch_normalized, axis=2, keepdims=True) < 0.9
    generated = generated * (1 - sketch_mask * 0.3) + sketch_normalized * sketch_mask * 0.3
    
    # Weichzeichnen für realistischeren Look
    generated_pil = Image.fromarray((generated * 255).astype(np.uint8))
    generated_pil = generated_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    generated = np.array(generated_pil).astype(np.float32) / 255.0
    
    return np.clip(generated, 0, 1)

# Bild zu Base64 konvertieren
def image_to_base64(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    image = image.resize((280, 280), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Numpy Array zu Base64 für generierte Bilder
def numpy_to_base64(image_array, size=(512, 512)):
    if len(image_array.shape) == 2:
        image = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
    else:
        image = Image.fromarray((image_array * 255).astype(np.uint8))
    
    if size:
        image = image.resize(size, Image.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# 20 zufällige Fashion Items auswählen
@st.cache_data
def select_random_fashion_items():
    x_all, y_all = load_fashion_mnist()
    
    if x_all is None:
        return []
    
    random_indices = random.sample(range(len(x_all)), 20)
    
    items = []
    for i, idx in enumerate(random_indices):
        image = x_all[idx]
        label = y_all[idx]
        category = FASHION_CLASSES[label]
        
        brands = ["StyleCo", "FashionHub", "TrendWear", "UrbanStyle", "ClassicMode", 
                 "ModernLook", "ChicWear", "ElegantStyle", "CasualFit", "PremiumWear"]
        prices = ["25€", "35€", "45€", "55€", "65€", "75€", "85€", "95€", "105€", "115€"]
        
        item = {
            "id": i + 1,
            "name": f"{category} #{idx}",
            "brand": random.choice(brands),
            "price": random.choice(prices),
            "category": category,
            "description": f"Stilvolle {category.lower()} aus der Fashion-MNIST Kollektion",
            "image_data": image_to_base64(image),
            "image_array": image,
            "original_index": idx,
            "label": label,
            "timestamp": datetime.now().isoformat()
        }
        items.append(item)
    
    return items

# Session State initialisieren
def init_session_state():
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'liked_items' not in st.session_state:
        st.session_state.liked_items = []
    if 'disliked_items' not in st.session_state:
        st.session_state.disliked_items = []
    if 'fashion_items' not in st.session_state:
        with st.spinner("Lade Fashion-MNIST Datensatz..."):
            st.session_state.fashion_items = select_random_fashion_items()
    if 'all_time_favorites' not in st.session_state:
        st.session_state.all_time_favorites = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "swipe"
    if 'selected_for_generation' not in st.session_state:
        st.session_state.selected_for_generation = []
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'pix2pix_model' not in st.session_state:
        with st.spinner("Lade Pix2Pix Turbo Modell..."):
            st.session_state.pix2pix_model = load_pix2pix_model()

def like_item():
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.liked_items.append(current_item)
        if not any(fav['original_index'] == current_item['original_index'] for fav in st.session_state.all_time_favorites):
            st.session_state.all_time_favorites.append(current_item)
        st.session_state.current_index += 1

def dislike_item():
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.disliked_items.append(current_item)
        st.session_state.current_index += 1

def reset_session():
    st.session_state.current_index = 0
    st.session_state.liked_items = []
    st.session_state.disliked_items = []
    with st.spinner("Lade neue Fashion-MNIST Bilder..."):
        st.session_state.fashion_items = select_random_fashion_items()

def render_swipe_tab():
    """Rendert den Swipe-Tab"""
    items = st.session_state.fashion_items
    current_idx = st.session_state.current_index
    total_items = len(items)
    
    if not items:
        st.error("Fehler beim Laden der Fashion-MNIST Daten.")
        return
    
    # Progress Bar
    progress = current_idx / total_items if total_items > 0 else 0
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress * 100}%"></div>
    </div>
    <p style="text-align: center; color: #666;">
        {current_idx} von {total_items} Artikeln angeschaut
    </p>
    """, unsafe_allow_html=True)
    
    # Statistiken
    liked_count = len(st.session_state.liked_items)
    disliked_count = len(st.session_state.disliked_items)
    
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">❤️ {liked_count}</span>
            <span class="stat-label">Gefällt mir</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">👎 {disliked_count}</span>
            <span class="stat-label">Nicht interessiert</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">⭐ {len(st.session_state.all_time_favorites)}</span>
            <span class="stat-label">Alle Favoriten</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if current_idx >= total_items:
        st.markdown("## 🎉 Session beendet!")
        st.success(f"Du hast alle {total_items} Fashion-MNIST Artikel durchgesehen!")
        
        if st.button("🔄 Neue Session starten", type="primary", use_container_width=True):
            reset_session()
            st.rerun()
            
    else:
        current_item = items[current_idx]
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown(f"""
            <div class="fashion-card" id="fashion-card-{current_idx}">
                <div class="category-tag">{current_item['category']}</div>
                <div class="item-name">{current_item['name']}</div>
                <div class="item-brand">{current_item['brand']}</div>
                <div class="item-price">{current_item['price']}</div>
                <p style="color: #666; line-height: 1.6;">{current_item['description']}</p>
                <div class="fashion-image">
                    <img src="{current_item['image_data']}" style="width: 100%; border-radius: 10px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
        
        with col2:
            if st.button("👎 Nicht interessiert", type="secondary", use_container_width=True):
                dislike_item()
                st.rerun()
        
        with col4:
            if st.button("❤️ Gefällt mir!", type="primary", use_container_width=True):
                like_item()
                st.rerun()

def render_generate_tab():
    """Rendert den Generate-Tab mit echtem Pix2Pix Modell"""
    st.markdown("## 🎨 Dein personalisiertes Fashion-Design generieren")
    
    # Zeige Modell-Status
    if PIX2PIX_AVAILABLE and st.session_state.pix2pix_model is not None:
        st.success("✅ Pix2Pix Turbo Modell geladen - Realistische Bildgenerierung verfügbar!")
    elif PIX2PIX_AVAILABLE:
        st.warning("⚠️ Pix2Pix Modell konnte nicht geladen werden - Verwende verbesserte Mock-Generierung")
    else:
        st.info("ℹ️ Pix2Pix Modell nicht verfügbar - Verwende verbesserte Mock-Generierung")
    
    if not st.session_state.all_time_favorites:
        st.info("👗 Sammle erst einige Favoriten durch Swipen, um dein personalisiertes Fashion-Design zu generieren!")
        return
    
    st.markdown("""
    <div class="generate-container">
        <h3>✨ Erstelle dein einzigartiges Fashion-Design</h3>
        <p>Wähle deine Lieblings-Styles aus und lass die KI ein personalisiertes Design für dich erstellen!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Favoriten-Auswahl
    st.markdown("### 1️⃣ Wähle deine Lieblings-Styles")
    st.markdown("*Klicke auf die Bilder, um sie für die Generierung auszuwählen (max. 5)*")
    
    # Grid für Favoriten
    cols = st.columns(5)
    for idx, item in enumerate(st.session_state.all_time_favorites[:20]):
        col_idx = idx % 5
        with cols[col_idx]:
            is_selected = item['original_index'] in [s['original_index'] for s in st.session_state.selected_for_generation]
            
            if st.button(
                f"{'✅' if is_selected else '⬜'}", 
                key=f"select_{item['original_index']}",
                help=f"{item['category']} - Klicken zum {'Abwählen' if is_selected else 'Auswählen'}"
            ):
                if is_selected:
                    st.session_state.selected_for_generation = [
                        s for s in st.session_state.selected_for_generation 
                        if s['original_index'] != item['original_index']
                    ]
                else:
                    if len(st.session_state.selected_for_generation) < 5:
                        st.session_state.selected_for_generation.append(item)
                    else:
                        st.warning("Maximal 5 Items können ausgewählt werden!")
                st.rerun()
            
            st.image(item['image_data'], caption=item['category'], use_container_width=True)
    
    # Zeige ausgewählte Items
    if st.session_state.selected_for_generation:
        st.markdown(f"### 2️⃣ Ausgewählte Styles ({len(st.session_state.selected_for_generation)}/5)")
        
        selected_categories = [item['category'] for item in st.session_state.selected_for_generation]
        category_text = ", ".join(selected_categories)
        st.markdown(f"**Kategorien:** {category_text}")
        
        # Style-Optionen
        st.markdown("### 3️⃣ Style-Anpassungen")
        
        col1, col2 = st.columns(2)
        with col1:
            style_mood = st.selectbox(
                "Stimmung",
                ["Modern & Trendy", "Klassisch & Elegant", "Sportlich & Casual", 
                 "Vintage & Retro", "Minimalistisch", "Extravagant"]
            )
        
        with col2:
            color_scheme = st.selectbox(
                "Farbschema",
                ["Natürliche Töne", "Monochrom", "Pastellfarben", 
                 "Kräftige Farben", "Dunkle Töne", "Bunte Mischung"]
            )
        
        # Erweiterte Optionen
        st.markdown("### 4️⃣ Erweiterte Optionen")
        
        col1, col2 = st.columns(2)
        with col1:
            generation_quality = st.selectbox(
                "Generierungs-Qualität",
                ["Standard", "Hohe Qualität", "Experimental"]
            )
        
        with col2:
            use_deterministic = st.checkbox("Deterministische Generierung", value=True, 
                                          help="Aktiviert für konsistente Ergebnisse")
        
        # Prompt erstellen
        prompt_base = f"A {style_mood.lower()} fashion design with {color_scheme.lower()}, inspired by {category_text}"
        
        # Erweiterte Prompt-Anpassungen
        if generation_quality == "Hohe Qualität":
            prompt = f"High resolution, professional fashion photography, {prompt_base}, studio lighting, clean background, detailed textures"
        elif generation_quality == "Experimental":
            prompt = f"Artistic fashion illustration, {prompt_base}, creative interpretation, unique style"
        else:
            prompt = prompt_base
        
        st.markdown("### 5️⃣ Generiere dein Design")
        
        # Zeige Preview der Skizze
        with st.expander("🖊️ Skizze-Vorschau anzeigen"):
            preview_sketch = create_realistic_fashion_sketch(st.session_state.selected_for_generation)
            st.image(preview_sketch, caption="Diese Skizze wird als Basis für die Generierung verwendet", 
                    use_container_width=True)
        
        if st.button("🎨 Design generieren!", type="primary", use_container_width=True):
            with st.spinner("🎨 Dein personalisiertes Fashion-Design wird erstellt..."):
                progress_bar = st.progress(0)
                
                # Simuliere erweiterten Generierungsprozess
                steps = ["Lade Modell...", "Erstelle Skizze...", "Generiere Design...", "Finalisiere Bild..."]
                for i, step in enumerate(steps):
                    st.text(step)
                    for j in range(25):
                        progress_bar.progress((i * 25 + j + 1))
                        import time
                        time.sleep(0.01)
                
                # Verwende echtes Pix2Pix Modell oder verbesserte Mock-Funktion
                generated_image = generate_fashion_with_pix2pix(
                    st.session_state.selected_for_generation,
                    prompt,
                    st.session_state.pix2pix_model
                )
                
                if generated_image is not None:
                    generation_data = {
                        'image': numpy_to_base64(generated_image),
                        'prompt': prompt,
                        'selected_items': st.session_state.selected_for_generation,
                        'timestamp': datetime.now().isoformat(),
                        'style_mood': style_mood,
                        'color_scheme': color_scheme,
                        'quality': generation_quality,
                        'deterministic': use_deterministic
                    }
                    st.session_state.generated_images.append(generation_data)
                    
                    st.markdown("""
                    <div class="generated-result">
                        <h3>🎉 Dein personalisiertes Fashion-Design!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(generated_image, caption="Dein generiertes Fashion-Design", use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📝 Design-Details")
                        st.markdown(f"**Stimmung:** {style_mood}")
                        st.markdown(f"**Farbschema:** {color_scheme}")
                        st.markdown(f"**Qualität:** {generation_quality}")
                        st.markdown(f"**Basiert auf:** {len(st.session_state.selected_for_generation)} Favoriten")
                        
                        # Download-Button
                        img_buffer = io.BytesIO()
                        img = Image.fromarray((generated_image * 255).astype(np.uint8))
                        img.save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="💾 Design herunterladen",
                            data=img_buffer,
                            file_name=f"fashion_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    if st.button("🔄 Neue Auswahl treffen", use_container_width=True):
                        st.session_state.selected_for_generation = []
                        st.rerun()
                else:
                    st.error("Fehler bei der Bildgenerierung. Bitte versuche es erneut.")
    else:
        st.info("Wähle mindestens ein Item aus deinen Favoriten aus, um ein Design zu generieren.")

def render_gallery_tab():
    """Rendert die Galerie der generierten Designs"""
    st.markdown("## 🖼️ Deine Fashion-Design Galerie")
    
    if not st.session_state.generated_images:
        st.info("Noch keine Designs generiert. Gehe zum 'Generate' Tab um dein erstes Design zu erstellen!")
        return
    
    st.markdown(f"### 📊 {len(st.session_state.generated_images)} generierte Designs")
    
    cols = st.columns(3)
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        col_idx = idx % 3
        with cols[col_idx]:
            st.image(gen_data['image'], use_container_width=True)
            st.caption(f"{gen_data['style_mood']} - {gen_data['color_scheme']}")
            st.text(f"Qualität: {gen_data.get('quality', 'Standard')}")
            st.text(f"Erstellt: {datetime.fromisoformat(gen_data['timestamp']).strftime('%d.%m.%Y %H:%M')}")
            
            if st.button(f"💾 Download", key=f"download_gallery_{idx}"):
                img_data = gen_data['image'].split(',')[1]
                img_bytes = base64.b64decode(img_data)
                
                st.download_button(
                    label="💾 Herunterladen",
                    data=img_bytes,
                    file_name=f"fashion_design_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key=f"download_btn_{idx}"
                )

def render_favorites_tab():
    """Zeigt die Favoriten in einer kompakten Übersicht"""
    st.markdown("### ⭐ Deine Favoriten-Sammlung")
    
    if not st.session_state.all_time_favorites:
        st.info("Noch keine Favoriten gesammelt. Swipe nach rechts um Favoriten zu sammeln!")
        return
    
    total_favorites = len(st.session_state.all_time_favorites)
    categories = [item['category'] for item in st.session_state.all_time_favorites]
    unique_categories = len(set(categories))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamt", total_favorites)
    with col2:
        st.metric("Kategorien", unique_categories)
    with col3:
        most_common = Counter(categories).most_common(1)[0][0]
        st.metric("Favorit", most_common)
    
    selected_category = st.selectbox(
        "Nach Kategorie filtern:",
        ["Alle"] + sorted(list(set(categories)))
    )
    
    if selected_category == "Alle":
        filtered_favorites = st.session_state.all_time_favorites
    else:
        filtered_favorites = [f for f in st.session_state.all_time_favorites if f['category'] == selected_category]
    
    st.markdown(f"**{len(filtered_favorites)} Items**")
    cols = st.columns(6)
    for idx, item in enumerate(filtered_favorites):
        with cols[idx % 6]:
            st.image(item['image_data'], caption=item['category'], use_container_width=True)
            if st.button("❌", key=f"remove_{item['original_index']}", help="Aus Favoriten entfernen"):
                st.session_state.all_time_favorites = [
                    f for f in st.session_state.all_time_favorites 
                    if f['original_index'] != item['original_index']
                ]
                st.rerun()

def main():
    init_session_state()
    
    st.markdown('<div class="main-header">👗 Fashion Swipe & Generate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Entdecke deinen Style und generiere einzigartige Fashion-Designs!</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Swipe", "🎨 Generate", "🖼️ Galerie", "⭐ Favoriten"])
    
    with tab1:
        render_swipe_tab()
    
    with tab2:
        render_generate_tab()
    
    with tab3:
        render_gallery_tab()
    
    with tab4:
        render_favorites_tab()
    
    # Sidebar mit erweiterten Infos
    with st.sidebar:
        st.markdown("## 🎯 Fashion Swipe & Generate")
        st.markdown("---")
        
        # Model Status
        if PIX2PIX_AVAILABLE and st.session_state.pix2pix_model is not None:
            st.success("🤖 Pix2Pix Turbo: Aktiv")
        else:
            st.info("🤖 Mock-Generierung: Aktiv")
        
        st.markdown("### 📊 Deine Statistiken")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("❤️ Likes", len(st.session_state.liked_items))
            st.metric("⭐ Favoriten", len(st.session_state.all_time_favorites))
        with col2:
            st.metric("👎 Dislikes", len(st.session_state.disliked_items))
            st.metric("🎨 Designs", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        st.markdown("### 💡 So funktioniert's")
        st.markdown("""
        1. **Swipe** durch Fashion-Items
        2. **Like** deine Favoriten (👍)
        3. **Wähle** Favoriten für die Generierung
        4. **Generiere** dein personalisiertes Design
        5. **Speichere** deine Kreationen
        """)
        
        if st.session_state.all_time_favorites:
            st.markdown("### 🏷️ Deine Top-Kategorien")
            categories = [item['category'] for item in st.session_state.all_time_favorites]
            category_counts = Counter(categories)
            for category, count in category_counts.most_common(3):
                st.markdown(f"**{category}**: {count}x")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Einstellungen")
        if st.button("🔄 Swipe-Session zurücksetzen"):
            reset_session()
            st.rerun()
        
        if st.button("🗑️ Alle generierten Designs löschen"):
            st.session_state.generated_images = []
            st.success("Alle Designs wurden gelöscht!")
            st.rerun()

if __name__ == "__main__":
    main()
