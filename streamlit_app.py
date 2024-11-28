import os
import subprocess
import sys
# Now import installed packages
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import traceback

# Rest of your script goes here


# Modify import to handle potential errors
try:
    from advanced_speech_recognition import AdvancedSpeechRecognitionModel
except Exception as e:
    st.error(f"Error importing model: {e}")
    st.error(traceback.format_exc())
    sys.exit(1)

class SpeechRecognitionApp:
    def __init__(self):
        """
        Initialize the Streamlit application for Speech Recognition
        """
        # Configure page
        st.set_page_config(
            page_title="Digit Speech Recognition", 
            page_icon="🎙️", 
            layout="wide"
        )
        
        # Initialize model state
        if 'model' not in st.session_state:
            st.session_state.model = None

    def run(self):
        """
        Main application runner
        """
        st.title("🎙️ Advanced Digit Speech Recognition")
        
        # Sidebar navigation
        page = st.sidebar.radio(
            "Navigation", 
            ["Train Model", "Predict Digit", "Model Info"]
        )
        
        # Page routing
        if page == "Train Model":
            self.train_model_page()
        elif page == "Predict Digit":
            self.predict_page()
        else:
            self.model_info_page()
    
    def train_model_page(self):
        """
        Model training page with enhanced logging and visualization
        """
        st.header("🚀 Model Training")
        
        # Dataset directory input
        data_dir = st.text_input(
            "Dataset Directory Path", 
            placeholder="Enter full path to speech dataset"
        )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Test Set Size", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05
            )
        with col2:
            epochs = st.slider(
                "Training Epochs", 
                min_value=10, 
                max_value=200, 
                value=50
            )
        
        # Detailed training logs toggle
        show_details = st.checkbox("Show Detailed Training Logs", value=False)
        
        # Train button
        if st.button("Start Training"):
            # Validate directory
            if not os.path.exists(data_dir):
                st.error("Invalid dataset directory!")
                return
            
            try:
                # Create and train model with verbose logging
                with st.spinner("Training in progress..."):
                    # Initialize model with verbose mode
                    speech_model = AdvancedSpeechRecognitionModel(data_dir, verbose=show_details)
                    
                    # Train model
                    history = speech_model.train(
                        test_size=test_size, 
                        epochs=epochs
                    )
                
                # Save model
                speech_model.save_model()
                
                # Store model reference in session state
                st.session_state.model = speech_model
                
                # Display training results
                st.success("Model Training Complete!")
                
                # Plot training history
                self._plot_training_history(history)
                
                # Optionally show detailed logs
                if show_details:
                    st.subheader("🔍 Training Details")
                    # Capture the output of display_training_details
                    import io
                    import sys
                    
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = io.StringIO()
                    
                    speech_model.display_training_details()
                    
                    sys.stdout = old_stdout
                    
                    st.text(captured_output.getvalue())
            
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.error(traceback.format_exc())
                
    def _plot_training_history(self, history):
        """
        Visualize training metrics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        st.pyplot(fig)
    
    def predict_page(self):
        """
        Digit prediction page
        """
        st.header("🔮 Digit Prediction")
        
        # Check if model is trained
        if st.session_state.model is None:
            st.warning("Train a model first!")
            return
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload WAV Audio", 
            type=['wav']
        )
        
        if uploaded_file is not None:
            # Save temporary file
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prediction button
            if st.button("Predict"):
                try:
                    # Make prediction
                    predicted_digit, confidence = st.session_state.model.predict("temp_audio.wav")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Digit", predicted_digit)
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.error(traceback.format_exc())
    
    def model_info_page(self):
        """
        Model information page
        """
        st.header("📊 Model Details")
        
        st.markdown("""
        ### Advanced Speech Recognition Model
        
        #### Key Features:
        - Convolutional Neural Network (CNN)
        - Spectral Feature Extraction
        - Data Augmentation
        - Adaptive Learning Techniques
        
        ### Architecture
        - Input Layer: 2D Spectrogram
        - 3 Convolutional Layers
        - Batch Normalization
        - Dropout Regularization
        - Dense Fully Connected Layers
        """)

def main():
    app = SpeechRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()
