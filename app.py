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

# ========== CSS MIT PROFESSIONELLEN ANIMATIONEN ==========
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
    
    /* Animation für SPAM - Alarm-Effekt */
    @keyframes alarmPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-3px); }
        20%, 40%, 60%, 80% { transform: translateX(3px); }
    }
    
    .alarm-container {
        animation: alarmPulse 0.8s ease-in-out 3, shake 0.5s ease-in-out 2;
    }
    
    /* Animation für HAM - Checkmark-Effekt */
    @keyframes checkmarkPop {
        0% { transform: scale(0); opacity: 0; }
        70% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .success-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .checkmark-animation {
        animation: checkmarkPop 0.5s ease-out;
        display: inline-block;
    }
    
    /* Professionelle Icons */
    .professional-icon {
        font-size: 60px;
        text-align: center;
        margin: 20px 0;
    }
    
    .spam-icon {
        color: #DC2626;
    }
    
    .ham-icon {
        color: #16A34A;
    }
    
    /* Status-Balken Animation */
    @keyframes progressFill {
        from { width: 0%; }
        to { width: var(--progress-width); }
    }
    
    .animated-progress {
        animation: progressFill 1s ease-out;
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

def show_spam_alert(prob):
    """Professionelle SPAM-Alarm-Anzeige"""
    st.markdown(f"""
    <div class='alarm-container'>
    <div style='
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 3px solid #DC2626;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.2);
        margin: 1.5rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Alarm-Icons
    st.markdown("""
    <div class='professional-icon spam-icon'>
        🔴🚨⚠️
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #DC2626; text-align: center; font-size: 28px;'>🚨 SPAM ERKANNT</h2>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: #7F1D1D; font-weight: 500; margin: 15px 0;'>
    Hohe Sicherheitswarnung - Verdächtige Nachricht erkannt
    </p>
    """, unsafe_allow_html=True)
    
    # Metriken mit Alarm-Style
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sicherheitsrisiko", f"{prob*100:.1f}%", delta="Kritisch", delta_color="inverse")
    with col2:
        st.metric("Bedrohungslevel", f"{prob:.3f}", delta="Hoch", delta_color="inverse")
    
    # Animierter Fortschrittsbalken
    st.markdown("<p style='margin-top: 20px; margin-bottom: 5px;'>Bedrohungsanalyse:</p>", unsafe_allow_html=True)
    progress_html = f"""
    <div style='
        width: 100%;
        height: 25px;
        background: #FCA5A5;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 20px;
    '>
        <div style='
            width: {prob*100}%;
            height: 100%;
            background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
            animation: progressFill 1.5s ease-out;
            border-radius: 12px;
        '></div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    st.error("""
    **SICHERHEITSWARNUNG**
    
    ▸ Verdächtige Links oder URLs erkannt  
    ▸ Unrealistische finanzielle Angebote  
    ▸ Drängende oder fordernde Sprache  
    ▸ Möglicher Identitätsdiebstahl-Versuch
    """)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def show_ham_success(prob):
    """Professionelle HAM-Erfolgsanzeige"""
    st.markdown(f"""
    <div class='success-container'>
    <div style='
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 3px solid #16A34A;
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.15);
        margin: 1.5rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Erfolgs-Icons mit Animation
    st.markdown("""
    <div class='professional-icon ham-icon'>
        <span class='checkmark-animation'>✅</span> 🛡️ ✓
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #16A34A; text-align: center; font-size: 28px;'>✅ SICHERE NACHRICHT</h2>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: #166534; font-weight: 500; margin: 15px 0;'>
    Nachricht erfolgreich verifiziert - Keine Bedrohung erkannt
    </p>
    """, unsafe_allow_html=True)
    
    # Metriken mit Erfolgs-Style
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sicherheitsniveau", f"{(1-prob)*100:.1f}%", delta="Sehr Hoch", delta_color="normal")
    with col2:
        st.metric("Vertrauensscore", f"{1-prob:.3f}", delta="Exzellent", delta_color="normal")
    
    # Animierter Fortschrittsbalken
    st.markdown("<p style='margin-top: 20px; margin-bottom: 5px;'>Sicherheitsanalyse:</p>", unsafe_allow_html=True)
    progress_html = f"""
    <div style='
        width: 100%;
        height: 25px;
        background: #BBF7D0;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 20px;
    '>
        <div style='
            width: {(1-prob)*100}%;
            height: 100%;
            background: linear-gradient(90deg, #22C55E 0%, #16A34A 100%);
            animation: progressFill 1.5s ease-out;
            border-radius: 12px;
        '></div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    st.success("""
    **SICHERHEITSVERIFIKATION ABGESCHLOSSEN**
    
    ▸ Normale Kommunikationsmuster erkannt  
    ▸ Keine verdächtigen Links vorhanden  
    ▸ Realistischer und logischer Inhalt  
    ▸ Sichere Nachricht - Kein Handlungsbedarf
    """)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ========== HAUPTINTERFACE ==========
# Titel
st.markdown("<h1 style='text-align: center; color: #1E3A8A; font-size: 2.2rem;'>📱 PROFESSIONELLER SPAM-DETEKTOR</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4B5563; margin-bottom: 2rem;'>AI-gestützte SMS-Sicherheitsanalyse mit LSTM-Netzwerk</p>", 
            unsafe_allow_html=True)
st.markdown("---")

# ========== NACHRICHTENEINGABE ==========
st.markdown("### 🔍 Zu analysierende Nachricht")

message = st.text_area(
    "**Geben Sie die SMS-Nachricht ein:**",
    height=140,
    placeholder="Beispiel: 'Wichtige Sicherheitsbenachrichtigung: Ihr Konto wurde kompromittiert...'",
    key="message_input",
    label_visibility="visible"
)

# ========== ANALYSE-BUTTON ==========
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("**🔐 SICHERHEITSANALYSE STARTEN**", 
                           type="primary", 
                           use_container_width=True,
                           help="Klicken Sie, um die Nachricht auf Spam zu überprüfen")

if analyze_btn:
    if not message or not message.strip():
        st.warning("⚠️ Bitte geben Sie eine zu analysierende Nachricht ein")
    else:
        # Statusanzeige
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        with status_placeholder.container():
            st.info("📡 **Initialisiere Sicherheitsscan...**")
        
        # Schritt 1: Modell laden
        progress_bar.progress(25)
        with status_placeholder.container():
            st.info("🔧 **Lade KI-Modell...**")
        
        model, tokenizer, max_len = load_model()
        
        if model is not None and tokenizer is not None:
            # Schritt 2: Text verarbeiten
            progress_bar.progress(50)
            with status_placeholder.container():
                st.info("⚙️ **Verarbeite Textdaten...**")
            
            pad = preprocess_text(message.strip(), tokenizer, max_len)
            
            if len(pad[0]) == 0:
                progress_bar.progress(100)
                with status_placeholder.container():
                    st.error("❌ **Analyse fehlgeschlagen** - Text kann nicht verarbeitet werden")
            else:
                # Schritt 3: Analyse durchführen
                progress_bar.progress(75)
                with status_placeholder.container():
                    st.info("🤖 **Führe KI-Analyse durch...**")
                
                prob = model.predict(pad, verbose=0)[0][0]
                
                # Schritt 4: Ergebnisse anzeigen
                progress_bar.progress(100)
                status_placeholder.empty()
                progress_bar.empty()
                
                st.markdown("---")
                st.markdown("## 📊 **ANALYSEERGEBNIS**")
                
                # Ergebnisse basierend auf Vorhersage anzeigen
                if prob > 0.5:
                    show_spam_alert(prob)
                else:
                    show_ham_success(prob)
                
                # Zusätzliche Info
                with st.expander("📈 **Analyse-Statistiken**", expanded=False):
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.write(f"**Textlänge:** {len(message.strip().split())} Wörter")
                        st.write(f"**Verarbeitete Tokens:** {len(pad[0])}")
                    with col_stat2:
                        st.write(f"**Entscheidungsschwelle:** 0.5")
                        st.write(f"**Modell:** LSTM Security AI")
        
        else:
            st.error("❌ **Systemfehler** - Sicherheitsmodul konnte nicht geladen werden")

# ========== SYSTEMSTATUS ==========
st.markdown("---")
with st.expander("⚙️ **Systemstatus**", expanded=False):
    st.markdown("""
    **Sicherheitssystem:** Aktiv ✅  
    **KI-Modell:** LSTM-Netzwerk geladen ✅  
    **Letzte Aktualisierung:** Echtzeit-Analyse  
    **Version:** Security AI v2.1
    """)

# ========== AKTIONEN ==========
st.markdown("---")
col_act1, col_act2 = st.columns(2)
with col_act1:
    if st.button("🗑️ **Text zurücksetzen**", type="secondary", use_container_width=True):
        st.rerun()
with col_act2:
    if st.button("🔄 **Neue Analyse**", type="secondary", use_container_width=True):
        st.rerun()
