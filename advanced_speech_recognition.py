import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class AdvancedSpeechRecognitionModel:
    def __init__(self, data_dir, max_pad_len=173, verbose=True):
        """
        Initialisation du modèle de reconnaissance vocale avancé avec logging amélioré
        
        Args:
            data_dir (str): Répertoire contenant les données d'entraînement
            max_pad_len (int): Longueur maximale pour le padding
            verbose (bool): Active les logs détaillés
        """
        self.data_dir = data_dir
        self.max_pad_len = max_pad_len
        self.model = None
        self.label_encoder = LabelEncoder()
        self.verbose = verbose
        
        # Logs pour suivre chaque étape
        self.training_logs = {
            'dataset_info': {},
            'preprocessing_steps': [],
            'training_steps': []
        }

    def extract_features(self, audio_path):
        """
        Extraction des caractéristiques spectrographiques avec augmentation
        
        Args:
            audio_path (str): Chemin du fichier audio
        
        Returns:
            np.array: Spectrogramme normalisé
        """
        try:
            # Charger l'audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convertir en mono si stéréo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Augmentation : ajout de bruit léger
            noise = np.random.normal(0, 0.005 * np.max(audio), audio.shape)
            audio = audio + noise
            
            # Générer le spectrogramme
            spectrogram = librosa.stft(audio)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
            
            # Normaliser et padding
            if spectrogram_db.shape[1] > self.max_pad_len:
                spectrogram_db = spectrogram_db[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - spectrogram_db.shape[1]
                spectrogram_db = np.pad(spectrogram_db, 
                                        ((0, 0), (0, pad_width)), 
                                        mode='constant')
            
            return spectrogram_db
        
        except Exception as e:
            print(f"Erreur lors du traitement de {audio_path}: {e}")
            return None

    def build_cnn_model(self, input_shape):
        """
        Builds a Convolutional Neural Network (CNN) for speech recognition.
        
        Args:
            input_shape (tuple): Shape of the input data (height, width, channels).
            
        Returns:
            keras.Model: Compiled CNN model.
        """
        model = models.Sequential()
        
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten and Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes
        
        # Compile the model
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model

    def log_step(self, stage, message):
        """
        Enregistre les étapes importantes avec un message
        
        Args:
            stage (str): Étape du processus
            message (str): Description de l'étape
        """
        if self.verbose:
            print(f"[{stage}] {message}")
        
        if stage in ['preprocessing', 'training']:
            self.training_logs[f'{stage}_steps'].append(message)
    
    def visualize_dataset_distribution(self, y):
        """
        Visualisation de la distribution des classes
        
        Args:
            y (np.array): Labels du dataset
        """
        plt.figure(figsize=(10, 5))
        
        # Distribution des classes
        class_counts = np.bincount(y)
        
        if SEABORN_AVAILABLE:
            # Si seaborn est disponible, utiliser seaborn
            sns.barplot(x=range(len(class_counts)), y=class_counts)
        else:
            # Sinon utiliser matplotlib standard
            plt.bar(range(len(class_counts)), class_counts)
        
        plt.title('Distribution des Classes')
        plt.xlabel('Classe (Chiffre)')
        plt.ylabel('Nombre d\'échantillons')
        plt.xticks(range(len(class_counts)), range(10))
        plt.tight_layout()
        plt.show()
    
    def load_dataset(self, verbose=True):
        """
        Chargement et préparation du dataset avec logging détaillé
        
        Returns:
            tuple: Données X et labels y
        """
        features = []
        labels = []
        
        # Parcourir tous les sous-répertoires (d0, d1, etc.)
        for digit_dir in sorted(os.listdir(self.data_dir)):
            full_path = os.path.join(self.data_dir, digit_dir)
            
            if os.path.isdir(full_path):
                digit = int(digit_dir[1])  # Extraire le chiffre du nom du répertoire
                
                # Traiter chaque fichier audio
                for audio_file in os.listdir(full_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(full_path, audio_file)
                        feature = self.extract_features(audio_path)
                        
                        if feature is not None:
                            features.append(feature)
                            labels.append(digit)
        
        # Convertir en numpy arrays
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        
        # Reshape pour CNN (height, width, channels)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Logs détaillés
        self.training_logs['dataset_info'] = {
            'total_samples': len(X),
            'class_distribution': list(np.bincount(y)),
            'feature_shape': X[0].shape
        }
        
        if verbose:
            print("\n📊 Statistiques du Dataset:")
            print(f"Nombre total d'échantillons : {len(X)}")
            print(f"Répartition des classes : {np.bincount(y)}")
            
            # Visualisation de la distribution
            try:
                self.visualize_dataset_distribution(y)
            except Exception as e:
                print(f"Erreur lors de la visualisation : {e}")
        
        return X, y
    
    def train(self, test_size=0.2, epochs=100):
        """
        Entraînement du modèle avec suivi détaillé des étapes
        
        Args:
            test_size (float): Proportion de données de test
            epochs (int): Nombre d'époques d'entraînement
        """
        # Étape 1: Préparation des données
        self.log_step('preprocessing', 'Chargement du dataset')
        X, y = self.load_dataset()
        
        # Étape 2: Normalisation
        self.log_step('preprocessing', 'Normalisation des données')
        X = X / 255.0
        
        # Étape 3: Séparation train/test
        self.log_step('preprocessing', f'Séparation train/test (test_size={test_size})')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Étape 4: Construction du modèle
        self.log_step('training', 'Construction de l\'architecture CNN')
        tf.keras.backend.clear_session()
        self.model = self.build_cnn_model(X_train.shape[1:])
        
        # Étape 5: Configuration des callbacks
        self.log_step('training', 'Configuration des callbacks d\'entraînement')
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=20, 
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=10, 
            min_lr=0.000001
        )
        
        # Étape 6: Entraînement
        self.log_step('training', f'Début de l\'entraînement ({epochs} époques)')
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Étape 7: Évaluation finale
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        self.log_step('training', f'Précision finale : {test_accuracy * 100:.2f}%')
        
        # Ajout des informations d'entraînement aux logs
        self.training_logs['training_info'] = {
            'best_val_accuracy': max(history.history['val_accuracy']),
            'best_val_loss': min(history.history['val_loss']),
            'final_test_accuracy': test_accuracy
        }
        
        # Visualisation
        self._plot_training_history(history)
        
        return history
    
    def display_training_details(self):
        """
        Affiche un résumé détaillé du processus de formation
        """
        print("\n🔍 Détails Complets de l'Entraînement:\n")
        
        # Informations sur le dataset
        print("📊 Informations sur le Dataset:")
        for key, value in self.training_logs['dataset_info'].items():
            print(f"   {key}: {value}")
        
        print("\n🚧 Étapes de Prétraitement:")
        for step in self.training_logs['preprocessing_steps']:
            print(f"   - {step}")
        
        print("\n🏋️ Étapes d'Entraînement:")
        for step in self.training_logs['training_steps']:
            print(f"   - {step}")
        
        print("\n📈 Résultats d'Entraînement:")
        if 'training_info' in self.training_logs:
            for key, value in self.training_logs['training_info'].items():
                print(f"   {key}: {value}")