import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== SEITENEINRICHTUNG ==========
st.set_page_config(
    page_title="Spam-SMS-Erkenner",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CSS ==========
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    
    .stTextArea textarea {
        font-size: 16px;
        min-height: 150px;
    }
    
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    
    .spam-alert {
        color: #DC2626;
        font-size: 24px;
        text-align: center;
        margin: 10px 0;
    }
    
    .ham-success {
        color: #16A34A;
        font-size: 24px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODELL LADEN ==========
@st.cache_resource
def load_model():
    """Lädt das LSTM-Modell und Tokenizer"""
    try:
        model = keras.models.load_model("spam_model.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, 40
    except Exception as e:
        st.error(f"Fehler beim Laden: {str(e)}")
        return None, None, None

# ========== HILFSFUNKTIONEN ==========
def preprocess_text(text, tokenizer, max_len):
    """Textverarbeitung für das Modell"""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")
    return pad

# ========== HAUPTINTERFACE ==========
# Titel
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📱 Spam SMS Erkennung</h1>", 
            unsafe_allow_html=True)
st.markdown("---")

# ========== NACHRICHTENEINGABE ==========
st.markdown("### 📝 Nachricht eingeben")

message = st.text_area(
    "**Geben Sie hier Ihre SMS-Nachricht ein:**",
    height=150,
    placeholder="Beispiel: 'URGENT! Your bank account has been suspended. Verify now...'",
    key="message_input",
    label_visibility="visible"
)

# ========== ANALYSE-BUTTON ==========
if st.button("**🔍 NACHRICHT ANALYSIEREN**", type="primary", use_container_width=True):
    if not message or not message.strip():
        st.warning("Bitte geben Sie eine Nachricht ein")
    else:
        with st.spinner("Lade Modell..."):
            model, tokenizer, max_len = load_model()
        
        if model is not None and tokenizer is not None:
            with st.spinner("Analysiere Nachricht..."):
                pad = preprocess_text(message.strip(), tokenizer, max_len)
                
                if len(pad[0]) == 0:
                    st.error("❌ Nachricht kann nicht analysiert werden")
                else:
                    prob = model.predict(pad, verbose=0)[0][0]
                    
                    # Ergebnisse anzeigen
                    st.markdown("---")
                    st.markdown("## 📊 Analyseergebnis")
                    
                    if prob > 0.5:
                        # SPAM
                        st.markdown("""
                        <div style='
                            background-color: #FEF2F2; 
                            padding: 2rem; 
                            border-radius: 10px; 
                            border: 3px solid #DC2626;
                            margin: 1rem 0;
                        '>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<h2 style='color: #DC2626; text-align: center;'>🚨 SPAM ERKANNT 🚨</h2>", 
                                  unsafe_allow_html=True)
                        
                        # Gefahrensymbol
                        st.markdown("""
                        <div style='text-align: center; font-size: 40px; margin: 20px 0;'>
                            ⚠️🚫🔥
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metriken
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Konfidenz", f"{prob*100:.1f}%")
                        with col2:
                            st.metric("Vorhersagewert", f"{prob:.3f}")
                        
                        st.progress(float(prob))
                        
                        st.error("""
                        **⚠️ WARNUNG: SPAM-NACHRICHT ERKANNT!**
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    else:
                        # HAM
                        st.markdown("""
                        <div style='
                            background-color: #F0FDF4; 
                            padding: 2rem; 
                            border-radius: 10px; 
                            border: 3px solid #16A34A;
                            margin: 1rem 0;
                        '>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<h2 style='color: #16A34A; text-align: center;'>✅ KEIN SPAM</h2>", 
                                  unsafe_allow_html=True)
                        
                        # Balloons nur für HAM
                        st.balloons()
                        
                        # Metriken
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Konfidenz", f"{(1-prob)*100:.1f}%")
                        with col2:
                            st.metric("Vorhersagewert", f"{prob:.3f}")
                        
                        st.progress(float(1 - prob))
                        
                        st.success("""
                        **✓ SICHER: Normale Nachricht**
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.error("Modell konnte nicht geladen werden")

# ========== TEXT LÖSCHEN BUTTON ==========
st.markdown("---")
if st.button("🗑️ Text löschen", type="secondary", use_container_width=True):
    st.rerun()
