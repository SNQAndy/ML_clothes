import streamlit as st
import pandas as pd
import random
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Fashion Swipe",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS für besseres Styling + verbesserte Swipe Funktionalität
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
        -webkit-user-select: none;
        -ms-user-select: none;
    }
    
    .fashion-card:active {
        cursor: grabbing;
    }
    
    .fashion-card.dragging {
        transition: none;
        cursor: grabbing;
    }
    
    .fashion-card.removed-left {
        transform: translateX(-150%) rotate(-30deg) !important;
        opacity: 0 !important;
    }
    
    .fashion-card.removed-right {
        transform: translateX(150%) rotate(30deg) !important;
        opacity: 0 !important;
    }
    
    .swipe-indicator {
        position: absolute;
        top: 20px;
        font-size: 3rem;
        font-weight: bold;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 10;
        pointer-events: none;
    }
    
    .swipe-indicator.left {
        left: 20px;
        color: #ff4757;
    }
    
    .swipe-indicator.right {
        right: 20px;
        color: #2ed573;
    }
    
    .fashion-card.show-left .swipe-indicator.left {
        opacity: 1;
    }
    
    .fashion-card.show-right .swipe-indicator.right {
        opacity: 1;
    }
    
    .item-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .item-brand {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    
    .item-price {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF6B9D;
        margin-bottom: 1rem;
    }
    
    .category-tag {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    
    .stat-item {
        display: inline-block;
        margin: 0 20px;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
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
    
    .swipe-instructions {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4);
        color: white;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 1rem;
    }
    
    .swipe-arrows {
        font-size: 1.5rem;
        margin: 10px 0;
    }
    
    .fashion-image {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        background: white;
        padding: 10px;
        pointer-events: none;
    }
    
    .fashion-image img {
        pointer-events: none;
        -webkit-user-drag: none;
        -khtml-user-drag: none;
        -moz-user-drag: none;
        -o-user-drag: none;
        user-drag: none;
    }
    
    /* Favoriten-Liste Styling */
    .favorites-container {
        background: linear-gradient(135deg, #FF6B9D 0%, #4ECDC4 100%);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
    
    .favorites-header {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .favorite-item {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.2s ease;
    }
    
    .favorite-item:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.15);
    }
    
    .favorite-image {
        width: 80px;
        height: 80px;
        border-radius: 10px;
        margin-right: 15px;
        object-fit: cover;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .favorite-details {
        flex: 1;
    }
    
    .favorite-name {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .favorite-category {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 5px;
    }
    
    .favorite-price {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .favorite-actions {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    
    .remove-btn {
        background: rgba(255,0,0,0.2);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: background 0.2s ease;
    }
    
    .remove-btn:hover {
        background: rgba(255,0,0,0.4);
    }
    
    .empty-favorites {
        text-align: center;
        padding: 40px 20px;
        opacity: 0.8;
        font-size: 1.1rem;
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

# Fashion-MNIST Daten laden
@st.cache_data
def load_fashion_mnist():
    try:
        # Fashion-MNIST laden
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Alle Daten zusammenführen
        x_all = np.concatenate([x_train, x_test])
        y_all = np.concatenate([y_train, y_test])
        
        return x_all, y_all
    except Exception as e:
        st.error(f"Fehler beim Laden von Fashion-MNIST: {e}")
        return None, None

# Bild zu Base64 konvertieren
def image_to_base64(image_array):
    # Normalisieren und zu PIL Image konvertieren
    image = Image.fromarray(image_array.astype(np.uint8))
    
    # Bild vergrößern für bessere Darstellung
    image = image.resize((280, 280), Image.NEAREST)
    
    # Zu Base64 konvertieren
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

# 10 zufällige Fashion Items auswählen
@st.cache_data
def select_random_fashion_items():
    x_all, y_all = load_fashion_mnist()
    
    if x_all is None:
        return []
    
    # 10 zufällige Indizes auswählen
    random_indices = random.sample(range(len(x_all)), 10)
    
    items = []
    for i, idx in enumerate(random_indices):
        image = x_all[idx]
        label = y_all[idx]
        category = FASHION_CLASSES[label]
        
        # Zufällige Preise und Marken generieren
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
            "original_index": idx,
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
    if 'session_started' not in st.session_state:
        st.session_state.session_started = datetime.now()
    if 'all_time_favorites' not in st.session_state:
        st.session_state.all_time_favorites = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "swipe"

# Dynamische Favoriten-Liste Funktionen
def add_to_favorites(item):
    """Fügt ein Item zur dauerhaften Favoriten-Liste hinzu"""
    # Prüfe ob Item bereits in Favoriten ist
    if not any(fav['original_index'] == item['original_index'] for fav in st.session_state.all_time_favorites):
        favorite_item = item.copy()
        favorite_item['added_timestamp'] = datetime.now().isoformat()
        favorite_item['session_id'] = id(st.session_state.session_started)
        st.session_state.all_time_favorites.append(favorite_item)

def remove_from_favorites(original_index):
    """Entfernt ein Item aus der Favoriten-Liste"""
    st.session_state.all_time_favorites = [
        fav for fav in st.session_state.all_time_favorites 
        if fav['original_index'] != original_index
    ]

def get_favorites_by_category():
    """Gruppiert Favoriten nach Kategorien"""
    favorites_by_category = {}
    for fav in st.session_state.all_time_favorites:
        category = fav['category']
        if category not in favorites_by_category:
            favorites_by_category[category] = []
        favorites_by_category[category].append(fav)
    return favorites_by_category

def export_favorites_as_json():
    """Exportiert Favoriten als JSON"""
    return json.dumps(st.session_state.all_time_favorites, indent=2, ensure_ascii=False)

def export_favorites_as_csv():
    """Exportiert Favoriten als CSV DataFrame"""
    if not st.session_state.all_time_favorites:
        return pd.DataFrame()
    
    # Erstelle DataFrame ohne Bilddaten (zu groß für CSV)
    export_data = []
    for fav in st.session_state.all_time_favorites:
        export_data.append({
            'name': fav['name'],
            'brand': fav['brand'],
            'price': fav['price'],
            'category': fav['category'],
            'description': fav['description'],
            'original_index': fav['original_index'],
            'timestamp': fav.get('timestamp', ''),
            'added_timestamp': fav.get('added_timestamp', ''),
            'session_id': fav.get('session_id', '')
        })
    
    return pd.DataFrame(export_data)

def like_item():
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.liked_items.append(current_item)
        # Füge zu dauerhaften Favoriten hinzu
        add_to_favorites(current_item)
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
    st.session_state.session_started = datetime.now()

def render_swipe_tab():
    """Rendert den Swipe-Tab"""
    items = st.session_state.fashion_items
    current_idx = st.session_state.current_index
    total_items = len(items)
    
    if not items:
        st.error("Fehler beim Laden der Fashion-MNIST Daten. Bitte installieren Sie TensorFlow: `pip install tensorflow`")
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
    
    # Hauptinhalt
    if current_idx >= total_items:
        # Session beendet
        st.markdown("## 🎉 Session beendet!")
        st.success(f"Du hast alle {total_items} Fashion-MNIST Artikel durchgesehen!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❤️ Session Favoriten")
            if st.session_state.liked_items:
                for item in st.session_state.liked_items:
                    st.markdown(f"**{item['name']}** ({item['category']}) - {item['price']}")
            else:
                st.info("Keine Favoriten in dieser Session")
        
        with col2:
            st.markdown("### 👎 Nicht interessant")
            if st.session_state.disliked_items:
                for item in st.session_state.disliked_items:
                    st.markdown(f"**{item['name']}** ({item['category']}) - {item['price']}")
            else:
                st.info("Alle Artikel haben dir gefallen!")
        
        # Kategorie-Analyse
        if st.session_state.liked_items:
            st.markdown("### 📊 Deine Lieblingskategorien")
            liked_categories = [item['category'] for item in st.session_state.liked_items]
            category_counts = pd.Series(liked_categories).value_counts()
            
            for category, count in category_counts.items():
                st.markdown(f"**{category}**: {count} mal geliked")
        
        # Reset Button
        if st.button("🔄 Neue Session starten", type="primary", use_container_width=True):
            reset_session()
            st.rerun()
            
    else:
        # Swipe Instructions
        st.markdown("""
        <div class="swipe-instructions">
            <div class="swipe-arrows">← 👎 Swipe zum Bewerten 👍 →</div>
            <strong>Nach links wischen:</strong> Nicht interessiert &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Nach rechts wischen:</strong> Gefällt mir!
        </div>
        """, unsafe_allow_html=True)
        
        # Aktueller Artikel
        current_item = items[current_idx]
        
        # Artikel anzeigen
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown(f"""
            <div class="fashion-card" id="fashion-card-{current_idx}">
                <div class="swipe-indicator left">👎</div>
                <div class="swipe-indicator right">👍</div>
                <div class="category-tag">{current_item['category']}</div>
                <div class="item-name">{current_item['name']}</div>
                <div class="item-brand">{current_item['brand']}</div>
                <div class="item-price">{current_item['price']}</div>
                <p style="color: #666; line-height: 1.6;">{current_item['description']}</p>
                <div class="fashion-image">
                    <img src="{current_item['image_data']}" style="width: 100%; border-radius: 10px;" alt="{current_item['name']}" draggable="false">
                </div>
                <p style='text-align: center; color: #888; margin-top: 10px;'>Fashion-MNIST Index: {current_item['original_index']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sichtbare Buttons
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #888;'>Oder nutze die Buttons:</p>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
        
        with col2:
            if st.button("👎 Nicht interessiert", type="secondary", use_container_width=True, key=f"dislike_{current_idx}"):
                dislike_item()
                st.rerun()
        
        with col4:
            if st.button("❤️ Gefällt mir!", type="primary", use_container_width=True, key=f"like_{current_idx}"):
                like_item()
                st.rerun()
    
    # Verbessertes JavaScript für Swipe-Funktionalität (basierend auf testAnwendung.html)
    st.markdown(f"""
    <script>
    (function() {{
        let currentCard = null;
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        let offsetX = 0;
        let offsetY = 0;
        const swipeThreshold = 50;
        const rotationThreshold = 20;
        
        function initSwipe() {{
            currentCard = document.querySelector('#fashion-card-{current_idx}');
            if (!currentCard) return;
            
            // Entferne alte Event Listener
            currentCard.removeEventListener('touchstart', handleTouchStart);
            currentCard.removeEventListener('touchmove', handleTouchMove);
            currentCard.removeEventListener('touchend', handleTouchEnd);
            currentCard.removeEventListener('mousedown', handleMouseDown);
            
            // Touch Events
            currentCard.addEventListener('touchstart', handleTouchStart, {{ passive: false }});
            currentCard.addEventListener('touchmove', handleTouchMove, {{ passive: false }});
            currentCard.addEventListener('touchend', handleTouchEnd, {{ passive: false }});
            
            // Mouse Events für Desktop
            currentCard.addEventListener('mousedown', handleMouseDown);
        }}
        
        function handleTouchStart(e) {{
            if (!currentCard) return;
            e.preventDefault();
            isDragging = true;
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
            currentCard.classList.add('dragging');
        }}
        
        function handleTouchMove(e) {{
            if (!isDragging || !currentCard) return;
            e.preventDefault();
            
            const currentX = e.touches[0].clientX;
            const currentY = e.touches[0].clientY;
            offsetX = currentX - startX;
            offsetY = currentY - startY;
            
            updateCardPosition();
        }}
        
        function handleTouchEnd(e) {{
            if (!isDragging || !currentCard) return;
            e.preventDefault();
            isDragging = false;
            currentCard.classList.remove('dragging');
            
            handleSwipeEnd();
        }}
        
        function handleMouseDown(e) {{
            if (!currentCard) return;
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            currentCard.classList.add('dragging');
            e.preventDefault();
            
            // Event Listener für mousemove und mouseup auf document
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }}
        
        function handleMouseMove(e) {{
            if (!isDragging || !currentCard) return;
            
            const currentX = e.clientX;
            const currentY = e.clientY;
            offsetX = currentX - startX;
            offsetY = currentY - startY;
            
            updateCardPosition();
        }}
        
        function handleMouseUp(e) {{
            if (!isDragging || !currentCard) return;
            isDragging = false;
            currentCard.classList.remove('dragging');
            
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            
            handleSwipeEnd();
        }}
        
        function updateCardPosition() {{
            if (!currentCard) return;
            
            const rotate = offsetX / rotationThreshold;
            currentCard.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) rotate(${{rotate}}deg)`;
            
            // Zeige Swipe-Indikatoren
            currentCard.classList.remove('show-left', 'show-right');
            if (Math.abs(offsetX) > 50) {{
                if (offsetX > 0) {{
                    currentCard.classList.add('show-right');
                }} else {{
                    currentCard.classList.add('show-left');
                }}
            }}
        }}
        
        function handleSwipeEnd() {{
            if (!currentCard) return;
            
            if (Math.abs(offsetX) > swipeThreshold) {{
                if (offsetX > 0) {{
                    // Swipe Right - Like
                    currentCard.classList.add('removed-right');
                    setTimeout(() => {{
                        triggerLike();
                    }}, 300);
                }} else {{
                    // Swipe Left - Dislike  
                    currentCard.classList.add('removed-left');
                    setTimeout(() => {{
                        triggerDislike();
                    }}, 300);
                }}
            }} else {{
                // Zurück zur Originalposition
                currentCard.style.transform = 'translate(0,0) rotate(0deg)';
                currentCard.classList.remove('show-left', 'show-right');
            }}
            
            offsetX = 0;
            offsetY = 0;
        }}
        
        function triggerLike() {{
            // Finde den Like-Button über seinen Text
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {{
                if (button.textContent.includes('Gefällt mir!') && button.closest('[data-testid="column"]')) {{
                    button.click();
                }}
            }});
        }}
        
        function triggerDislike() {{
            // Finde den Dislike-Button über seinen Text
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {{
                if (button.textContent.includes('Nicht interessiert') && button.closest('[data-testid="column"]')) {{
                    button.click();
                }}
            }});
        }}
        
        // Initialisiere Swipe
        setTimeout(initSwipe, 100);
        
        // Re-initialisiere bei Änderungen
        const observer = new MutationObserver(function(mutations) {{
            setTimeout(initSwipe, 100);
        }});
        
        observer.observe(document.body, {{
            childList: true,
            subtree: true
        }});
    }})();
    </script>
    """, unsafe_allow_html=True)

def render_favorites_tab():
    """Rendert den Favoriten-Tab"""
    st.markdown("## ⭐ Deine Favoriten-Sammlung")
    
    if not st.session_state.all_time_favorites:
        st.markdown("""
        <div class="favorites-container">
            <div class="empty-favorites">
                <h3>🤷‍♀️ Noch keine Favoriten</h3>
                <p>Swipe nach rechts um deine ersten Favoriten zu sammeln!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Favoriten-Statistiken
    total_favorites = len(st.session_state.all_time_favorites)
    favorites_by_category = get_favorites_by_category()
    most_liked_category = max(favorites_by_category.keys(), key=lambda k: len(favorites_by_category[k])) if favorites_by_category else "Keine"
    
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">⭐ {total_favorites}</span>
            <span class="stat-label">Gesamte Favoriten</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">📂 {len(favorites_by_category)}</span>
            <span class="stat-label">Kategorien</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">👑 {most_liked_category}</span>
            <span class="stat-label">Top Kategorie</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Export-Optionen
    st.markdown("### 📤 Export-Optionen")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Als JSON exportieren", use_container_width=True):
            json_data = export_favorites_as_json()
            st.download_button(
                label="💾 JSON herunterladen",
                data=json_data,
                file_name=f"fashion_favorites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Als CSV exportieren", use_container_width=True):
            csv_data = export_favorites_as_csv()
            if not csv_data.empty:
                st.download_button(
                    label="💾 CSV herunterladen",
                    data=csv_data.to_csv(index=False),
                    file_name=f"fashion_favorites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col3:
        if st.button("🗑️ Alle löschen", use_container_width=True):
            if st.button("⚠️ Wirklich löschen?", type="secondary"):
                st.session_state.all_time_favorites = []
                st.rerun()
    
    # Filter-Optionen
    st.markdown("### 🔍 Filter")
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        category_filter = st.selectbox(
            "Nach Kategorie filtern:",
            ["Alle"] + list(favorites_by_category.keys()),
            key="category_filter"
        )
    
    with filter_col2:
        sort_option = st.selectbox(
            "Sortieren nach:",
            ["Neueste zuerst", "Älteste zuerst", "Kategorie A-Z", "Kategorie Z-A"],
            key="sort_option"
        )
    
    # Favoriten filtern und sortieren
    filtered_favorites = st.session_state.all_time_favorites.copy()
    
    if category_filter != "Alle":
        filtered_favorites = [fav for fav in filtered_favorites if fav['category'] == category_filter]
    
    # Sortieren
    if sort_option == "Neueste zuerst":
        filtered_favorites.sort(key=lambda x: x.get('added_timestamp', ''), reverse=True)
    elif sort_option == "Älteste zuerst":
        filtered_favorites.sort(key=lambda x: x.get('added_timestamp', ''))
    elif sort_option == "Kategorie A-Z":
        filtered_favorites.sort(key=lambda x: x['category'])
    elif sort_option == "Kategorie Z-A":
        filtered_favorites.sort(key=lambda x: x['category'], reverse=True)
    
    # Favoriten anzeigen
    st.markdown(f"### 💖 Deine Favoriten ({len(filtered_favorites)} Artikel)")
    
    if not filtered_favorites:
        st.info("Keine Favoriten mit den aktuellen Filtereinstellungen gefunden.")
        return
    
    # Favoriten in schöner Liste anzeigen
    for i, favorite in enumerate(filtered_favorites):
        st.markdown(f"""
        <div class="favorite-item">
            <img src="{favorite['image_data']}" class="favorite-image" alt="{favorite['name']}">
            <div class="favorite-details">
                <div class="favorite-name">{favorite['name']}</div>
                <div class="favorite-category">📂 {favorite['category']} | 🏷️ {favorite['brand']}</div>
                <div class="favorite-price">{favorite['price']}</div>
                <small style="opacity: 0.7;">
                    Hinzugefügt: {datetime.fromisoformat(favorite.get('added_timestamp', datetime.now().isoformat())).strftime('%d.%m.%Y %H:%M')}
                </small>
            </div>
            <div class="favorite-actions">
                <button class="remove-btn" onclick="removeFavorite({favorite['original_index']})">
                    🗑️ Entfernen
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Versteckter Remove-Button für jedes Item
        if st.button("🗑️", key=f"remove_fav_{favorite['original_index']}", help="Aus Favoriten entfernen"):
            remove_from_favorites(favorite['original_index'])
            st.rerun()

def render_analytics_tab():
    """Rendert den Analytics-Tab"""
    st.markdown("## 📊 Deine Fashion-Analyse")
    
    if not st.session_state.all_time_favorites:
        st.info("Sammle erst einige Favoriten um deine persönliche Fashion-Analyse zu sehen!")
        return
    
    # Kategorie-Analyse
    favorites_by_category = get_favorites_by_category()
    
    st.markdown("### 📈 Kategorie-Verteilung")
    category_data = []
    for category, items in favorites_by_category.items():
        category_data.append({"Kategorie": category, "Anzahl": len(items)})
    
    if category_data:
        df_categories = pd.DataFrame(category_data)
        st.bar_chart(df_categories.set_index("Kategorie"))
    
    # Top-Statistiken
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Kategorien")
        sorted_categories = sorted(favorites_by_category.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (category, items) in enumerate(sorted_categories[:5], 1):
            percentage = (len(items) / len(st.session_state.all_time_favorites)) * 100
            st.markdown(f"**{i}. {category}**: {len(items)} Artikel ({percentage:.1f}%)")
    
    with col2:
        st.markdown("### 💰 Preisverteilung")
        prices = [fav['price'] for fav in st.session_state.all_time_favorites]
        price_counts = pd.Series(prices).value_counts().head(5)
        for price, count in price_counts.items():
            st.markdown(f"**{price}**: {count} mal geliked")
    
    # Timeline-Analyse
    st.markdown("### 📅 Like-Timeline")
    timeline_data = []
    for fav in st.session_state.all_time_favorites:
        if 'added_timestamp' in fav:
            date = datetime.fromisoformat(fav['added_timestamp']).date()
            timeline_data.append({"Datum": date, "Likes": 1})
    
    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        df_timeline = df_timeline.groupby("Datum").sum().reset_index()
        st.line_chart(df_timeline.set_index("Datum"))
    
    # Detailanalyse
    st.markdown("### 🔍 Detailanalyse")
    total_favorites = len(st.session_state.all_time_favorites)
    
    # Marken-Analyse
    brands = [fav['brand'] for fav in st.session_state.all_time_favorites]
    brand_counts = pd.Series(brands).value_counts()
    most_liked_brand = brand_counts.index[0] if len(brand_counts) > 0 else "Keine"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Gesamte Likes", total_favorites)
    with col2:
        st.metric("📂 Verschiedene Kategorien", len(favorites_by_category))
    with col3:
        st.metric("🏷️ Top Marke", most_liked_brand)
    with col4:
        avg_likes_per_day = len(st.session_state.all_time_favorites) / max(1, (datetime.now() - st.session_state.session_started).days or 1)
        st.metric("📊 Likes/Tag", f"{avg_likes_per_day:.1f}")

def main():
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">👗 Fashion Swipe</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Entdecke deinen Style mit Fashion-MNIST Datensatz!</div>', unsafe_allow_html=True)
    
    # Tab-Auswahl mit gestylten Streamlit-Buttons
    st.markdown("""
    <style>
    /* Tab Button Container - gezielt das erste Element nach dem Header */
    div[data-testid="stHorizontalBlock"]:first-of-type {
        background: #f8f9fa;
        border-radius: 25px;
        padding: 5px;
        margin: 20px 0;
    }
    
    /* Tab Buttons Styling */
    div[data-testid="stHorizontalBlock"]:first-of-type button {
        background: transparent;
        border: none;
        border-radius: 20px;
        padding: 12px 20px;
        font-weight: 500;
        color: #666;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stHorizontalBlock"]:first-of-type button:hover {
        background: #e9ecef;
        color: #333;
    }
    
    /* Active Tab Styling */
    """ + ("""
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(1) button {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255,107,157,0.3) !important;
    }
    """ if st.session_state.current_tab == "swipe" else "") + ("""
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(2) button {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255,107,157,0.3) !important;
    }
    """ if st.session_state.current_tab == "favorites" else "") + ("""
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(3) button {
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255,107,157,0.3) !important;
    }
    """ if st.session_state.current_tab == "analytics" else "") + """
    </style>
    """, unsafe_allow_html=True)
    
    # Tab Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Swipe", key="tab_swipe", use_container_width=True):
            st.session_state.current_tab = "swipe"
            st.rerun()
    with col2:
        if st.button(f"⭐ Favoriten ({len(st.session_state.all_time_favorites)})", key="tab_favorites", use_container_width=True):
            st.session_state.current_tab = "favorites"
            st.rerun()
    with col3:
        if st.button("📊 Analyse", key="tab_analytics", use_container_width=True):
            st.session_state.current_tab = "analytics"
            st.rerun()
    
    # Tab-Inhalt rendern
    if st.session_state.current_tab == "swipe":
        render_swipe_tab()
    elif st.session_state.current_tab == "favorites":
        render_favorites_tab()
    elif st.session_state.current_tab == "analytics":
        render_analytics_tab()
    
    # Sidebar für zusätzliche Features
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")
        
        if st.button("🔄 Session zurücksetzen"):
            reset_session()
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 📊 Live-Statistiken")
        
        if 'session_started' in st.session_state:
            time_elapsed = datetime.now() - st.session_state.session_started
            st.metric("⏱️ Zeit", f"{time_elapsed.seconds // 60}min {time_elapsed.seconds % 60}s")
        
        st.metric("🎯 Session bewertet", st.session_state.current_index)
        st.metric("❤️ Session Likes", len(st.session_state.liked_items))
        st.metric("👎 Session Dislikes", len(st.session_state.disliked_items))
        st.metric("⭐ Alle Favoriten", len(st.session_state.all_time_favorites))
        
        if st.session_state.current_index > 0:
            like_rate = (len(st.session_state.liked_items) / st.session_state.current_index) * 100
            st.metric("📈 Session Like-Rate", f"{like_rate:.1f}%")
        
        st.markdown("---")
        st.markdown("## 🎯 Favoriten-Schnellzugriff")
        
        if st.session_state.all_time_favorites:
            # Letzte 3 Favoriten anzeigen
            st.markdown("**Zuletzt hinzugefügt:**")
            recent_favorites = sorted(
                st.session_state.all_time_favorites, 
                key=lambda x: x.get('added_timestamp', ''), 
                reverse=True
            )[:3]
            
            for fav in recent_favorites:
                st.markdown(f"• {fav['name']} ({fav['category']})")
        else:
            st.info("Noch keine Favoriten gesammelt")
        
        st.markdown("---")
        st.markdown("## 📱 Swipe Anleitung")
        st.markdown("""
        **Desktop:** Klicken und ziehen
        
        **Mobile:** Touch und wischen
        
        - **Nach links:** 👎 Nicht interessiert
        - **Nach rechts:** 👍 Gefällt mir
        
        *Gelikte Items werden automatisch zu deinen Favoriten hinzugefügt!*
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Fashion-MNIST Info")
        st.markdown(f"""
        **Datensatz:** 70.000 Bilder
        
        **Kategorien:** {len(FASHION_CLASSES)}
        
        **Aktuelle Session:** {len(st.session_state.fashion_items)} zufällige Bilder
        
        **Bildgröße:** 28x28 Pixel (vergrößert auf 280x280)
        """)
        
        # Kategorie-Übersicht
        with st.expander("📋 Alle Kategorien"):
            for class_id, class_name in FASHION_CLASSES.items():
                st.markdown(f"• {class_name}")

    # JavaScript für Remove-Funktion
    st.markdown("""
    <script>
    function removeFavorite(originalIndex) {
        // Trigger entsprechenden Remove-Button
        const removeButtons = document.querySelectorAll('[data-testid="stButton"] button');
        removeButtons.forEach(button => {
            if (button.getAttribute('key') === `remove_fav_${originalIndex}`) {
                button.click();
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
