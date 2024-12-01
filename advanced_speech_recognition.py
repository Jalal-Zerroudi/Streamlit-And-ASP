# advanced_speech_recognition.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config

class AdvancedSpeechRecognitionModel:
    def __init__(self, config=Config):
        """
        Enhanced initialization with centralized configuration

        Args:
            config (class): Configuration class with model parameters
        """
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

        # Validate and create necessary paths
        config.validate_paths()

        # GPU Configuration
        self._configure_gpu()

    def _configure_gpu(self):
        """
        Advanced GPU configuration with multiple optimization strategies
        """
        if self.config.PERFORMANCE['use_gpu']:
            physical_devices = tf.config.list_physical_devices('GPU')

            if physical_devices:
                try:
                    # Enable memory growth
                    if self.config.PERFORMANCE['memory_growth']:
                        for device in physical_devices:
                            tf.config.experimental.set_memory_growth(device, True)

                    # Mixed precision training
                    if self.config.PERFORMANCE['mixed_precision']:
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')

                    print(f"✅ GPU Optimization Enabled: {len(physical_devices)} GPU(s)")

                except Exception as e:
                    print(f"❌ GPU Configuration Error: {e}")
            else:
                print("❗ No GPU detected. Falling back to CPU.")

    def extract_features(self, audio_path):
        """
        Advanced feature extraction with multiple augmentation techniques

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np.array: Enhanced spectrogram features
        """
        try:
            # Load audio with enhanced reading
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono and normalize
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Multiple augmentation strategies
            augmented_audio = self._apply_audio_augmentations(audio)

            # Generate spectrogram with more advanced parameters
            spectrogram = librosa.stft(augmented_audio,
                                       n_fft=2048,
                                       hop_length=512)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=np.max)

            # Dynamic padding
            max_pad_len = self.config.MODEL['max_pad_len']
            if spectrogram_db.shape[1] > max_pad_len:
                spectrogram_db = spectrogram_db[:, :max_pad_len]
            else:
                pad_width = max_pad_len - spectrogram_db.shape[1]
                spectrogram_db = np.pad(spectrogram_db,
                                        ((0, 0), (0, pad_width)),
                                        mode='constant')

            return spectrogram_db

        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def _apply_audio_augmentations(self, audio):
        """
        Advanced audio augmentation techniques

        Args:
            audio (np.array): Input audio signal

        Returns:
            np.array: Augmented audio signal
        """
        # Noise injection
        noise = np.random.normal(0, 0.005 * np.max(audio), audio.shape)
        audio_with_noise = audio + noise

        # Slight time stretching
        stretch_factor = np.random.uniform(0.9, 1.1)
        stretched_audio = librosa.effects.time_stretch(audio_with_noise, rate=stretch_factor)

        return stretched_audio

    def build_model(self):
        """
        Advanced CNN architecture with residual connections and advanced regularization

        Returns:
            tf.keras.Model: Compiled neural network model
        """
        # Modify input shape creation
        input_shape = (self.config.MODEL['max_pad_len'], self.config.MODEL['max_pad_len'], 1)

        model = models.Sequential([
            layers.Input(shape=(None, None, 1)),
            # Input layer with explicitly defined input shape
            layers.Input(shape=input_shape),

            # First Convolutional Block with Residual Connection
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Global Features
            layers.GlobalAveragePooling2D(),

            # Dense Layers with Advanced Regularization
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),

            # Output Layer
            layers.Dense(10, activation='softmax')
        ])

        # Advanced Optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.TRAINING['learning_rate']
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self):
        """
        Enhanced training method with advanced callbacks and logging
        """
        # Load and preprocess dataset
        X, y = self._load_dataset()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TRAINING['test_size'],
            stratify=y,
            random_state=42
        )

        # Build model
        self.model = self.build_model()

        # Callbacks
        callbacks_list = [
            # Early Stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.TRAINING['early_stopping_patience'],
                restore_best_weights=True
            ),
            # Learning Rate Reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10
            )
        ]

        # Optional TensorBoard logging
        if self.config.LOGGING['tensorboard_logs']:
            log_dir = os.path.join(
                self.config.LOGGING['log_dir'],
                f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks_list.append(tensorboard_callback)

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config.TRAINING['epochs'],
            batch_size=self.config.TRAINING['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return self.history

    def save_model(self, save_path=None):
        """
        Save trained model with comprehensive metadata
        """
        if save_path is None:
            save_path = self.config.MODEL['save_path']

        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, 'speech_model.h5')

        self.model.save(model_path)
        print(f"✅ Model saved successfully at {model_path}")

    def _load_dataset(self):
        """
        Load and preprocess the audio dataset for training

        Returns:
            tuple: X (features), y (labels)
        """
        X = []
        y = []

        # Print the full base path to help debug
        print(f"Looking for dataset in: {self.config.DATASET['base_path']}")

        # Check if base dataset directory exists
        if not os.path.exists(self.config.DATASET['base_path']):
            raise FileNotFoundError(f"Dataset base directory not found: {self.config.DATASET['base_path']}")

        # Iterate through dataset subdirectories
        for label, subdir in enumerate(self.config.DATASET['subdirs']):
            subdir_path = os.path.join(self.config.DATASET['base_path'], subdir)

            # Check if directory exists
            if not os.path.exists(subdir_path):
                print(f"Warning: Directory {subdir_path} not found")
                continue

            # Iterate through audio files in the subdirectory
            audio_files = [f for f in os.listdir(subdir_path) if f.endswith(self.config.DATASET['file_extension'])]

            if not audio_files:
                print(f"No audio files found in {subdir_path}")
                continue

            for filename in audio_files:
                file_path = os.path.join(subdir_path, filename)

                try:
                    # Extract features
                    features = self.extract_features(file_path)

                    if features is not None:
                        # Reshape features for model input
                        features = features.reshape((*features.shape, 1))
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Provide more detailed error message
        if len(X) == 0:
            raise ValueError(f"""
            No audio files could be processed in the dataset.
            Please ensure:
            1. Dataset directory exists: {self.config.DATASET['base_path']}
            2. Subdirectories exist: {self.config.DATASET['subdirs']}
            3. .wav files are present and valid in these subdirectories
            4. Feature extraction is working correctly
            """)

        print(f"Loaded dataset: {len(X)} samples across {len(np.unique(y))} classes")

        return X, y
    

    def load_model(self, model_path):
        """
        Load a pre-trained model from an H5 file
        """
        self.model = tf.keras.models.load_model(model_path)
