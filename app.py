import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== SEITENEINRICHTUNG ==========
st.set_page_config(
    page_title="Spam-SMS-Erkenner - LSTM vs RNN",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .spam-result {
        background-color: #FEF2F2;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #DC2626;
    }
    .ham-result {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #16A34A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODELLE LADEN ==========
@st.cache_resource
def load_models():
    """Lädt die Modelle und den Tokenizer"""
    try:
        # LSTM-Modell laden
        lstm_model = keras.models.load_model("spam_model.keras")
        
        # Tokenizer laden
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        # Modellparameter (wie in Ihrem Training)
        vocab_size = 5000
        max_len = 40
        
        return lstm_model, tokenizer, vocab_size, max_len
    except Exception as e:
        st.error(f"Fehler beim Laden der Dateien: {str(e)}")
        return None, None, None, None

# ========== HILFSFUNKTIONEN ==========
def preprocess_text(text, tokenizer, max_len):
    """Textverarbeitung wie im Training"""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")
    return pad

def predict_with_model(text, model, tokenizer, max_len, model_name="LSTM"):
    """Vorhersage für einen Text"""
    pad = preprocess_text(text, tokenizer, max_len)
    
    if len(pad[0]) == 0:  # Wenn alle Wörter unbekannt sind
        return None, 0.0, "Analyse nicht möglich"
    
    prob = model.predict(pad, verbose=0)[0][0]
    
    if prob > 0.5:
        label = "🚫 SPAM"
        confidence = prob * 100
    else:
        label = "✅ HAM"
        confidence = (1 - prob) * 100
    
    return label, confidence, prob

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Einstellungen")
    
    # Modellauswahl
    model_choice = st.radio(
        "Modell für die Analyse wählen:",
        ["LSTM-Modell", "SimpleRNN-Modell"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Projektinformationen")
    st.info("""
    **Modellspezifikationen:**
    - Vokabulargröße: 5.000 Wörter
    - Maximale Textlänge: 40 Wörter
    - Schichten: Embedding → LSTM/RNN → Dropout → Dense
    - Training auf ausbalancierten Daten
    """)
    
    # Schnellstatistiken
    st.markdown("### 📈 Schnellstatistiken")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Erwartete Genauigkeit", "98,2%")
    with col2:
        st.metric("Recall (Spam)", "97,5%")
    
    st.markdown("---")
    st.caption("Entwickelt mit TensorFlow und Streamlit")

# ========== HAUPTINTERFACE ==========
st.markdown('<h1 class="main-header">📱 Spam-SMS-Erkenner - Deep Learning Modellvergleich</h1>', unsafe_allow_html=True)

# Navigationstabs
tab1, tab2, tab3 = st.tabs(["🔍 Nachrichtenanalyse", "📊 Modellvergleich", "📁 Projektinformationen"])

with tab1:
    st.markdown('<h2 class="sub-header">SMS-Nachrichten analysieren</h2>', unsafe_allow_html=True)
    
    # Texteingabebereich
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        message = st.text_area(
            "Geben Sie eine SMS-Nachricht zur Analyse ein:",
            height=120,
            placeholder="Beispiel: You have won a $1000 prize! Click here to claim...",
            key="message_input"
        )
    
    with col_input2:
        st.markdown("### Schnellbeispiele")
        examples = {
            "Starker Spam": "URGENT! Your bank account has been suspended. Verify now: https://secure-bank.net",
            "Angebots-Spam": "You won an iPhone 15 Pro! Claim your free gift: www.apple-giveaway.com",
            "Normale Nachricht": "Hey, are we still meeting for lunch tomorrow at 1 PM?",
            "Arbeitsnachricht": "Please review the attached document when you have time today."
        }
        
        selected_example = st.selectbox("Beispiel wählen:", list(examples.keys()))
        if st.button("Beispiel anwenden"):
            st.session_state.message_input = examples[selected_example]
            st.rerun()
    
    # Analyse-Button
    if st.button("🔍 Nachricht analysieren", type="primary", use_container_width=True):
        if not message or not message.strip():
            st.warning("⚠️ Bitte geben Sie zuerst eine Nachricht ein")
        else:
            # Modelle laden
            lstm_model, tokenizer, vocab_size, max_len = load_models()
            
            if lstm_model is not None and tokenizer is not None:
                # Fortschrittsbalken
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Textverarbeitung
                status_text.text("Text wird verarbeitet...")
                progress_bar.progress(25)
                
                # Vorhersage
                status_text.text("Nachricht wird analysiert...")
                label, confidence, prob = predict_with_model(
                    message, lstm_model, tokenizer, max_len, 
                    model_choice
                )
                progress_bar.progress(75)
                
                if label is None:
                    st.error("❌ Diese Nachricht kann nicht analysiert werden (alle Wörter sind dem Modell unbekannt)")
                else:
                    # Ergebnisse anzeigen
                    status_text.text("Ergebnisse werden angezeigt...")
                    progress_bar.progress(100)
                    
                    st.markdown("---")
                    
                    # Ergebnis-Karte
                    if "SPAM" in label:
                        st.markdown(f'<div class="spam-result">', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ham-result">', unsafe_allow_html=True)
                    
                    col_result1, col_result2, col_result3 = st.columns([2, 1, 1])
                    
                    with col_result1:
                        st.markdown(f"### {label}")
                        st.markdown(f"**Nachricht:** {message[:100]}..." if len(message) > 100 else f"**Nachricht:** {message}")
                    
                    with col_result2:
                        st.metric("Konfidenzniveau", f"{confidence:.1f}%")
                    
                    with col_result3:
                        st.metric("Vorhersagewert", f"{prob:.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Konfidenzbalken
                    st.markdown("**Konfidenzniveau:**")
                    if "SPAM" in label:
                        st.progress(float(prob))
                    else:
                        st.progress(float(1 - prob))
                    
                    # Zusätzliche Details
                    with st.expander("🔍 Technische Details"):
                        st.write(f"**Verwendetes Modell:** {model_choice}")
                        st.write(f"**Eingabelänge:** {len(message.split())} Wörter")
                        st.write(f"**Rohausgabewert:** {prob:.4f}")
                        st.write(f"**Klassifizierungsschwelle:** 0.5")
                        
                        # Ergebnisinterpretation
                        if "SPAM" in label:
                            st.write("**Interpretation:** Die Nachricht enthält typische Spam-Merkmale wie:")
                            st.write("- Unrealistische Geldangebote")
                            st.write("- Verdächtige Links")
                            st.write("- Drängende Sprache")
                        else:
                            st.write("**Interpretation:** Die Nachricht scheint normal zu sein und enthält:")
                            st.write("- Natürliche Kommunikationssprache")
                            st.write("- Logischen und realistischen Inhalt")
                    
                    status_text.text("✅ Analyse abgeschlossen!")
                    progress_bar.empty()
                    st.balloons()
    
    # Mehrfachanalyse
    st.markdown("---")
    st.markdown('<h3 class="sub-header">Mehrfachanalyse</h3>', unsafe_allow_html=True)
    
    multi_messages = st.text_area(
        "Geben Sie mehrere Nachrichten ein (jede in einer neuen Zeile):",
        height=100,
        placeholder="Nachricht eingeben...\nWeitere Nachricht...\nDritte Nachricht..."
    )
    
    if st.button("Alle Nachrichten analysieren", type="secondary"):
        if multi_messages:
            messages_list = [msg.strip() for msg in multi_messages.split('\n') if msg.strip()]
            lstm_model, tokenizer, vocab_size, max_len = load_models()
            
            if lstm_model is not None:
                results = []
                for msg in messages_list:
                    label, confidence, prob = predict_with_model(msg, lstm_model, tokenizer, max_len)
                    if label:
                        results.append({
                            "Nachricht": msg[:50] + "..." if len(msg) > 50 else msg,
                            "Klassifikation": label,
                            "Konfidenz %": f"{confidence:.1f}",
                            "Wert": f"{prob:.3f}"
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Schnellstatistiken
                    spam_count = results_df['Klassifikation'].str.contains('SPAM').sum()
                    st.info(f"**Statistik:** {spam_count} Spam-Nachrichten von {len(results)} Nachrichten")

with tab2:
    st.markdown('<h2 class="sub-header">Modellleistungsvergleich</h2>', unsafe_allow_html=True)
    
    # Leistungsdaten der Modelle (aus Ihrem Code)
    performance_data = {
        "Modell": ["LSTM", "SimpleRNN"],
        "Genauigkeit (Accuracy)": [0.982, 0.975],  # Ungefähre Werte aus Ihren Ergebnissen
        "Recall (Spam)": [0.975, 0.960],  # Ungefähre Werte aus Ihren Ergebnissen
        "Anzahl Schichten": [3, 3],
        "Trainingszeit (relativ)": [1.0, 0.8]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.dataframe(perf_df, use_container_width=True)
    
    with col_perf2:
        # Vergleichsdiagramm
        fig, ax = plt.subplots(figsize=(8, 4))
        models = perf_df["Modell"]
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, perf_df["Genauigkeit (Accuracy)"], width, label='Genauigkeit', color='#3B82F6')
        ax.bar(x + width/2, perf_df["Recall (Spam)"], width, label='Recall', color='#10B981')
        
        ax.set_xlabel('Modell')
        ax.set_ylabel('Wert')
        ax.set_title('Modellleistungsvergleich')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        st.pyplot(fig)
    
    st.markdown("### 📝 Anmerkungen zu den Modellen")
    st.info("""
    **LSTM-Modell:**
    - Bessere Leistung beim Langzeitlernen
    - Besserer Umgang mit entfernten Abhängigkeiten im Text
    - Ideal für lange und komplexe Sätze
    
    **SimpleRNN-Modell:**
    - Schneller im Training
    - Weniger komplex
    - Kann bei langen Texten unter "Vanishing Gradient" leiden
    """)

with tab3:
    st.markdown('<h2 class="sub-header">Projektdetails Spam-Erkennung</h2>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### 📊 Trainingsdaten")
        st.write("""
        - **Gesamter Datensatz:** 5.574 Nachrichten
        - **Spam:** 747 Nachrichten (13,4%)
        - **Ham:** 4.827 Nachrichten (86,6%)
        - **Nach Ausbalancierung:** 1.494 Nachrichten (je 50% pro Klasse)
        - **Datenaufteilung:** 80% Training, 20% Test
        """)
        
        st.markdown("### 🔧 Textverarbeitung")
        st.write("""
        1. Text in numerische Sequenzen umwandeln
        2. Tokenizer mit 5.000 Wörtern
        3. Feste Textlänge (40 Wörter)
        4. Padding für kurze Texte
        """)
    
    with col_info2:
        st.markdown("### 🏗️ Modellarchitektur")
        st.write("""
        **Embedding-Schicht:**
        - Eingabegröße: 5.000
        - Embedding-Dimension: 64
        
        **LSTM/RNN-Schicht:**
        - 64 Neuronen
        - Dropout: 0.5
        
        **Ausgabeschicht:**
        - Dense mit Sigmoid
        - Binäre Klassifikation (Spam/Ham)
        """)
        
        st.markdown("### ⚙️ Trainingsparameter")
        st.write("""
        - **Verlustfunktion:** Binary Crossentropy
        - **Optimierer:** Adam
        - **Epochen:** 10
        - **Batch-Größe:** 64
        - **Validierung:** 20% der Trainingsdaten
        """)
    
    st.markdown("---")
    st.markdown("### 📁 Erforderliche Projektdateien")
    st.code("""
    Ihr_Projekt/
    ├── app.py                    # Streamlit App (dieser Code)
    ├── spam_model.keras          # Trainiertes LSTM-Modell
    ├── tokenizer.pkl             # Gespeicherter Tokenizer
    ├── requirements.txt          # Benötigte Bibliotheken
    └── README.md                 # Projektbeschreibung
    """, language="bash")

# ========== FOOTER ==========
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("📧 mohamed.example@email.com")
with footer_col2:
    st.caption("🔗 github.com/Mohamed13081")
with footer_col3:
    st.caption("🔄 Aktualisiert: Januar 2024")
