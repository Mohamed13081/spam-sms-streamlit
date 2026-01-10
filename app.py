import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow import keras

# Seiteneinrichtung
st.set_page_config(
    page_title="Spam-SMS-Erkenner",
    page_icon="📱",
    layout="centered"
)

# Titel und Beschreibung
st.title("📱 Spam-SMS-Erkenner")
st.markdown("Ermittelt, ob eine SMS-Nachricht **Spam** oder **ham** (normal) ist.")
st.markdown("---")

# ========== FUNKTIONEN ZUM LADEN DER DATEIEN ==========
@st.cache_resource
def model_laden():
    """Lädt das trainierte Keras-Modell"""
    try:
        # Direkter Pfad - funktioniert auf Streamlit Cloud
        model = keras.models.load_model("spam_model.keras")
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}")
        return None

@st.cache_resource
def tokenizer_laden():
    """Lädt den Tokenizer"""
    try:
        # Direkter Pfad - funktioniert auf Streamlit Cloud
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except FileNotFoundError:
        st.error("Datei 'tokenizer.pkl' nicht gefunden!")
        # Diagnose: Verfügbare Dateien anzeigen
        st.write("Verfügbare Dateien im aktuellen Verzeichnis:")
        for datei in os.listdir('.'):
            st.write(f"- {datei}")
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden des Tokenizers: {str(e)}")
        return None

# ========== LADE DAS MODELL UND DEN TOKENIZER ==========
with st.spinner("Lade Modell und Tokenizer..."):
    model = model_laden()
    tokenizer = tokenizer_laden()

# Falls Fehler beim Laden auftreten
if model is None or tokenizer is None:
    st.error("Konnte nicht alle erforderlichen Dateien laden. Bitte überprüfen Sie die Dateien.")
    st.stop()

# Erfolgsmeldung
st.success("✅ Modell und Tokenizer erfolgreich geladen!")

# ========== BENUTZEROBERFLÄCHE ==========
st.subheader("Nachricht analysieren")

# Texteingabe
nachricht = st.text_area(
    "Geben Sie eine SMS-Nachricht ein:",
    height=150,
    placeholder="Beispiel: 'Sie haben einen Preis gewonnen! Klicken Sie hier, um ihn abzuholen.'",
    help="Geben Sie eine SMS-Nachricht ein, um zu prüfen, ob es sich um Spam handelt."
)

# Analyse-Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analysieren_button = st.button(
        "🔍 Nachricht analysieren",
        type="primary",
        use_container_width=True
    )

# ========== ANALYSEFUNKTIONALITÄT ==========
if analysieren_button:
    if not nachricht.strip():
        st.warning("⚠️ Bitte geben Sie eine Nachricht ein!")
    else:
        try:
            # Vorverarbeitung der Nachricht
            sequenzen = tokenizer.texts_to_sequences([nachricht])
            
            if len(sequenzen[0]) == 0:
                st.error("❌ Nachricht kann nicht analysiert werden (unbekannte Wörter).")
            else:
                # Vorhersage
                with st.spinner("Analysiere Nachricht..."):
                    vorhersage = model.predict(np.array(sequenzen), verbose=0)[0][0]
                
                # Ergebnisse anzeigen
                st.markdown("---")
                st.subheader("📊 Analyseergebnis")
                
                # Spalten für die Anzeige
                ergebnis_col, konfidenz_col = st.columns(2)
                
                with ergebnis_col:
                    if vorhersage > 0.5:
                        st.error(f"**🚫 SPAM**")
                        st.markdown("Diese Nachricht ist wahrscheinlich Spam.")
                    else:
                        st.success(f"**✅ HAM (Normal)**")
                        st.markdown("Diese Nachricht ist wahrscheinlich legitim.")
                
                with konfidenz_col:
                    if vorhersage > 0.5:
                        konfidenz = vorhersage * 100
                    else:
                        konfidenz = (1 - vorhersage) * 100
                    
                    st.metric(
                        label="Konfidenzniveau",
                        value=f"{konfidenz:.1f}%"
                    )
                
                # Fortschrittsbalken
                st.progress(float(vorhersage))
                
                # Detaillierte Informationen
                with st.expander("🔍 Details zur Analyse"):
                    st.write(f"**Rohwert der Vorhersage:** {vorhersage:.4f}")
                    st.write(f"**Schwellenwert:** 0.5")
                    st.write(f"**Anzahl der Tokens:** {len(sequenzen[0])}")
                    
                    # Wahrscheinlichkeiten anzeigen
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Wahrscheinlichkeit Ham", f"{(1-vorhersage)*100:.1f}%")
                    with col_prob2:
                        st.metric("Wahrscheinlichkeit Spam", f"{vorhersage*100:.1f}%")
        
        except Exception as e:
            st.error(f"❌ Fehler bei der Analyse: {str(e)}")


    
    # Verfügbarkeit der Dateien überprüfen
    st.markdown("**📁 Verfügbare Dateien:**")
    dateien = os.listdir('.')
    for datei in dateien:
        st.write(f"- {datei}")



# Füge Nachricht zum Session State hinzu, falls noch nicht vorhanden
if 'nachricht' not in st.session_state:
    st.session_state.nachricht = ""
elif st.session_state.nachricht != "":
    # Falls eine Testnachricht ausgewählt wurde, in die Textarea einfügen
    nachricht = st.session_state.nachricht

# Fußzeile
st.markdown("---")
st.caption("Entwickelt mit TensorFlow und Streamlit | 🇩🇪 Deutsche Version")

# Debug-Informationen (nur im Entwicklungsmodus)
if st.sidebar.checkbox("Debug-Modus", False):
    st.sidebar.subheader("Debug-Informationen")
    st.sidebar.write(f"TensorFlow-Version: {tf.__version__}")
    st.sidebar.write(f"Streamlit-Version: {st.__version__}")
    st.sidebar.write(f"Aktuelles Verzeichnis: {os.getcwd()}")
    st.sidebar.write("Dateien im Verzeichnis:", os.listdir('.'))
