import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow import keras

# Seiteneinrichtung - sauber und einfach
st.set_page_config(
    page_title="Spam-SMS-Erkenner",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"  # Sidebar automatisch geschlossen
)

# Sidebar komplett ausblenden
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    .stButton > button {
        width: 100%;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODELL UND DATEN LADEN ==========
@st.cache_resource
def load_model():
    """Lädt das trainierte Modell"""
    try:
        model = keras.models.load_model("spam_model.keras")
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells")
        return None

@st.cache_resource
def load_tokenizer():
    """Lädt den Tokenizer"""
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Fehler beim Laden der Verarbeitungsdatei")
        return None

# ========== HAUPTOBERFLÄCHE ==========
# Titel
st.title("📱 Spam-SMS-Erkenner")
st.markdown("Geben Sie eine SMS-Nachricht ein, um zu prüfen, ob sie **Spam** oder **normal** ist")
st.markdown("---")

# Modelle laden
model = load_model()
tokenizer = load_tokenizer()

if model is None or tokenizer is None:
    st.error("Anwendung konnte nicht geladen werden. Bitte Dateien überprüfen.")
    st.stop()

# Nachrichteneingabe
st.subheader("Nachricht zur Analyse eingeben")

# Texteingabefeld
message = st.text_area(
    "**Geben Sie die SMS-Nachricht hier ein:**",
    height=150,
    placeholder="Beispiel: 'Sie haben einen Preis gewonnen! Klicken Sie hier' oder 'Hallo, treffen wir uns zum Mittagessen?'",
    label_visibility="visible",
    key="message_input"
)

# Analyse-Button
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    analyze_btn = st.button(
        "🔍 **Nachricht analysieren**",
        type="primary",
        use_container_width=True,
        key="analyze_button"
    )

# ========== NACHRICHTENANALYSE ==========
if analyze_btn:
    if not message or not message.strip():
        st.warning("⚠️ Bitte geben Sie zuerst eine Nachricht ein")
    else:
        try:
            # Textverarbeitung
            sequences = tokenizer.texts_to_sequences([message.strip()])
            
            if not sequences or len(sequences[0]) == 0:
                st.error("❌ Nachricht kann nicht analysiert werden (unbekannte Wörter)")
            else:
                # Vorhersage
                with st.spinner("Analyse läuft..."):
                    prediction = model.predict(np.array(sequences), verbose=0)[0][0]
                
                # Ergebnisse anzeigen
                st.markdown("---")
                st.subheader("📊 **Analyseergebnis**")
                
                # Ergebnis anzeigen
                if prediction > 0.5:
                    # SPAM
                    st.error(f"## 🚫 **SPAM-NACHRICHT**")
                    
                    # Konfidenz anzeigen
                    col_conf1, col_conf2 = st.columns([2, 1])
                    with col_conf1:
                        st.metric("Konfidenzniveau", f"{prediction*100:.1f}%")
                    with col_conf2:
                        st.metric("Schwellenwert", "> 50%")
                    
                    # Fortschrittsbalken
                    st.progress(float(prediction))
                    
                    # Warnhinweis
                    st.warning("""
                    **⚠️ ACHTUNG:** Diese Nachricht weist typische Spam-Merkmale auf:
                    • Unrealistische Geldangebote oder Gewinne
                    • Verdächtige Links oder URLs
                    • Aufforderung zur Preisgabe persönlicher Daten
                    • Drängender oder eiliger Ton
                    • Rechtschreib- oder Grammatikfehler
                    """)
                    
                else:
                    # HAM (normale Nachricht)
                    st.success(f"## ✅ **NORMALE NACHRICHT**")
                    
                    # Konfidenz anzeigen
                    col_conf1, col_conf2 = st.columns([2, 1])
                    with col_conf1:
                        st.metric("Konfidenzniveau", f"{(1-prediction)*100:.1f}%")
                    with col_conf2:
                        st.metric("Schwellenwert", "< 50%")
                    
                    # Fortschrittsbalken
                    st.progress(float(1 - prediction))
                    
                    # Bestätigungshinweis
                    st.info("""
                    **✓ SICHER:** Diese Nachricht scheint legitim zu sein:
                    • Persönliche oder geschäftliche Kommunikation
                    • Realistischer und logischer Inhalt
                    • Keine verdächtigen Links
                    • Keine Aufforderung zu sensiblen Daten
                    • Natürlicher Sprachfluss
                    """)
                
                # Detaillierte Informationen (optional)
                with st.expander("🔍 Technische Details anzeigen"):
                    st.write(f"**Vorhersagewert:** {prediction:.4f}")
                    st.write(f"**Entscheidungsschwelle:** 0.5")
                    st.write(f"**Anzahl erkannte Wörter:** {len(sequences[0])}")
                    
                    # Wahrscheinlichkeitsverteilung
                    st.write("**Wahrscheinlichkeiten:**")
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("Normal (Ham)", f"{(1-prediction)*100:.1f}%")
                    with prob_col2:
                        st.metric("Spam", f"{prediction*100:.1f}%")
                        
        except Exception as e:
            st.error(f"❌ Fehler während der Analyse")

# ========== OPTIONAL: SCHNELLTEST BUTTONS ==========
# Nur wenn gewünscht - können entfernt werden
st.markdown("---")
st.markdown("### 🧪 Schnelltest mit Beispielen")

test_col1, test_col2 = st.columns(2)

with test_col1:
    if st.button("Spam Beispiel", use_container_width=True):
        st.session_state.message_input = "Sie haben 1000€ gewonnen! Klicken Sie hier: http://gewinn.link"
        st.rerun()

with test_col2:
    if st.button("Normale Nachricht", use_container_width=True):
        st.session_state.message_input = "Hallo, wollen wir morgen um 15 Uhr im Café treffen?"
        st.rerun()

# ========== OPTIONAL: ZUSÄTZLICHE BEISPIELE ==========
with st.expander("Weitere Testbeispiele anzeigen"):
    examples = st.columns(2)
    
    with examples[0]:
        st.write("**Spam-Beispiele:**")
        if st.button("Bank-Spam", key="bank_spam"):
            st.session_state.message_input = "Ihr Bankkonto wurde gesperrt. Verifizieren Sie jetzt: https://bank-verify.net"
            st.rerun()
        if st.button("Gewinnspiel", key="gewinn_spam"):
            st.session_state.message_input = "GRATIS iPhone 15! Sie wurden ausgewählt. Jetzt abholen: win-apple.com"
            st.rerun()
    
    with examples[1]:
        st.write("**Normale Nachrichten:**")
        if st.button("Terminerinnerung", key="termin_norm"):
            st.session_state.message_input = "Erinnerung: Ihr Arzttermin ist morgen um 10:30 Uhr"
            st.rerun()
        if st.button("Persönliche Nachricht", key="pers_norm"):
            st.session_state.message_input = "Kannst du Milch auf dem Heimweg mitbringen? Danke!"
            st.rerun()

# ========== FUSSNOTE ==========
st.markdown("---")
st.caption("Entwickelt mit TensorFlow und Streamlit")

# Text löschen Button
if st.button("🗑️ Text löschen", type="secondary"):
    st.rerun()
