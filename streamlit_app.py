import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import traceback
from datetime import datetime
from typing import Dict, Optional

# Enhanced page configuration with more details
st.set_page_config(
    page_title="Automatic Speech Recognition", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_project_metadata() -> None:
    st.markdown("""
    <style>
    .project-metadata-card {
        background: linear-gradient(135deg, #2E3A59 0%, #c3cfe2 100%);
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        padding: 25px;
        display: flex;
        align-items: center;
        gap: 20px;
        transition: all 0.3s ease;
        max-width: 1000px;
        margin: 10px auto;
    }
    .project-metadata-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    .project-logo {
        width: 300px; /* Taille réduite de la largeur */
        max-width: 100px; /* Limite de largeur maximale */
        height: auto; /* Pour préserver le ratio d'aspect */
        border-radius: 12px;
        border: 3px solid #2c3e50;
        transition: transform 0.3s ease;
    }
    .project-logo:hover {
        transform: scale(1.05);
    }
    .project-details {
        flex-grow: 1;
    }
    .project-details h2 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .project-details .metadata-list {
        list-style-type: none;
        padding: 0;
    }
    .project-details .metadata-list li {
        margin-bottom: 10px;
        color: #34495e;
        display: flex;
        align-items: center;
        font-size: 1rem;
    }
    .project-details .metadata-list li .icon {
        margin-right: 15px;
        color: #3498db;
        font-size: 1.2rem;
        min-width: 30px;
        text-align: center;
    }
    .project-details .metadata-list li .label {
        font-weight: bold;
        min-width: 120px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="project-metadata-card">
        <img src="https://www.fsdm.usmba.ac.ma/Template/useful/loading/img/logo.png" alt="Project Logo" class="project-logo">
        <div class="project-details">
            <h2>🎙️ Speech Recognition Project</h2>
            <ul class="metadata-list">
                <li><span class="icon">🏫</span><span class="label">Institution:</span> Université Sidi Mohamed Ben Abdellah</li>
                <li><span class="icon">📚</span><span class="label">Faculty:</span> Sciences Dhar El Mehraz - Fès</li>
                <li><span class="icon">🎓</span><span class="label">Program:</span> Master BDSI</li>
                <li><span class="icon">👨‍🏫</span><span class="label">Supervisor:</span> Prof. Hassan SATORI</li>
                <li><span class="icon">👨‍💻</span><span class="label">Developer:</span> Jalal Zerroudi</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True )


def validate_project_metadata() -> None:
    """
    Detailed project metadata validation with expandable section
    """
    detailed_metadata = {
        "🔬 Research Domain": "Automatic Speech Recognition",
        "💻 Technologies": "TensorFlow, Python, Streamlit",
        "📅 Project Duration": "2023-2024",
        "🌐 Scope": "Advanced Machine Learning Application"
    }

    with st.expander("🔍 Extended Project Details"):
        for icon, detail in detailed_metadata.items():
            st.markdown(f"{icon} **{detail.split(':')[0]}:** {detail.split(':')[1] if ':' in detail else detail}")

# Modify import to handle potential errors

try:
    from advanced_speech_recognition import AdvancedSpeechRecognitionModel
except ImportError as e:
    st.error(f"Error importing model: {e}")
    st.error(traceback.format_exc())
    sys.exit(1)

class SpeechRecognitionApp:

    def __init__(self):
        """
        Advanced Initialization with Enhanced Error Handling
        """
        # Modern error handling for model import
        try:
            from advanced_speech_recognition import AdvancedSpeechRecognitionModel
        except ImportError as e:
            st.error(f"🚨 Model Import Error: {e}")
            st.stop()

        # Initialize session state with more comprehensive checks
        if 'model' not in st.session_state:
            st.session_state.model = None
            st.session_state.training_completed = False

    def run(self):
        """
        Enhanced Application Runner
        """
        # Display project metadata at the top
        display_project_metadata()
        
        # Modern navigation with icons and descriptions
        page_config = {
            "🚀 Train Model": {
                "function": self.train_model_page,
                "description": "\nPrepare and train your speech recognition model"
            },
            "🔮 Predict Digit": {
                "function": self.predict_page,
                "description": "\nTest your trained model with audio inputs"
            },
            "📊 Model Insights": {
                "function": self.model_info_page,
                "description": "\nExplore model architecture and performance"
            }
        }
        
        # Sidebar with improved navigation
        with st.sidebar:
            st.title("🎙️ Voice AI Dashboard")
            selected_page = st.radio(
                "Navigation", 
                list(page_config.keys()),
                format_func=lambda x: f"{x} "# or add - {page_config[x]['description']}
            )
        
        # Execute selected page function
        page_config[selected_page]['function']()

    def train_model_page(self):
        """
        Enhanced model training page with optimized UX flow
        """
        st.header("🚀 Model Training Studio")
        st.markdown("### Suivez les étapes pour former votre modèle.")

        # Étape 1 : Sélection du dataset
        st.subheader("1️ Sélectionnez le dataset")
        data_dir = st.text_input(
            "Chemin du Dataset",
            placeholder="Exemple : /chemin/vers/le/dossier",
            help="Indiquez le chemin complet vers le dossier contenant vos données d'entraînement."
        )
        if not data_dir:
            st.info("Veuillez fournir un chemin valide pour continuer.")

        # Étape 2 : Configurations d'entraînement
        st.subheader("2️ Configurez les paramètres d'entraînement")
        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Taille du jeu de test (%)",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Pourcentage des données utilisé pour la validation."
            )
            epochs = st.slider(
                "Nombre d'époques",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Nombre de passages complets à travers le dataset d'entraînement."
            )

        with col2:
            show_details = st.checkbox("Afficher les journaux détaillés", value=True)
            use_augmentation = st.checkbox("Activer l'augmentation des données", value=False)

        # Vérifications avant l'entraînement
        if st.button("🚀 Lancer l'entraînement", type="primary"):
            if not os.path.exists(data_dir):
                st.error("Le chemin fourni est invalide ou introuvable !")
                return

            try:
                with st.spinner("Entraînement en cours, veuillez patienter..."):
                    # Initialisation du modèle
                    speech_model = AdvancedSpeechRecognitionModel(
                        data_dir,
                        verbose=show_details,
                        data_augmentation=use_augmentation
                    )

                    # Lancement de l'entraînement avec suivi
                    history = speech_model.train(test_size=test_size, epochs=epochs)

                    # Sauvegarde du modèle
                    speech_model.save_model()
                    st.session_state.model = speech_model

                    # Succès
                    st.success("🎉 Entraînement terminé avec succès !")
                    st.balloons()

                    # Visualisation des métriques
                    self._plot_training_history(history)

            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")
                st.error(traceback.format_exc())




    def predict_page(self):
        """
        Modern digit prediction interface
        """
        st.header("🔮 Digit Prediction")
        
        # Model availability check
        if st.session_state.model is None:
            st.warning("Train a model first!")
            return
        
        # Drag and drop file uploader with preview
        uploaded_file = st.file_uploader(
            "Upload WAV File", 
            type=['wav'],
            help="Supported: .wav audio files with spoken digits"
        )
        
        if uploaded_file is not None:
            # Temporary file handling
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prediction with visual feedback
            if st.button("Predict Digit", type="primary"):
                try:
                    predicted_digit, confidence = st.session_state.model.predict("temp_audio.wav")
                    
                    # Modern result display
                    st.balloons()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Digit", predicted_digit, help="Neural network prediction")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%", help="Model's confidence in prediction")
                
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    def model_info_page(self):
        """
        Comprehensive model information page
        """
        st.header("📊 Model Insights")
        
        # Tabbed interface for information
        tab1, tab2, tab3 = st.tabs(["Overview", "Architecture", "Performance"])
        
        with tab1:
            st.markdown("""
            ### Advanced Voice Recognition System
            An intelligent neural network designed to recognize spoken digits 
            with high accuracy using deep learning techniques.
            
            #### Key Features
            - Real-time digit recognition
            - Advanced spectral feature extraction
            - Adaptive learning mechanisms
            """)
        
        with tab2:
            st.markdown("""
            ### Neural Network Architecture
            - **Input Layer:** 2D Spectrogram
            - **Convolutional Layers:** 3 layers with batch normalization
            - **Regularization:** Dropout techniques
            - **Output:** Digit classification
            """)
        
        with tab3:
            st.markdown("Performance metrics and evaluation coming soon!")

    def _plot_training_history(self, history):
        """
        Enhanced training visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy visualization
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss visualization
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        st.pyplot(fig)

def main():
    app = SpeechRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()
