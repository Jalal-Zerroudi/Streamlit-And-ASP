# Speech Recognition Digit Classifier

A Streamlit application for training, inspecting, and using a neural network that classifies spoken digits from WAV audio.

## Features

- Train a TensorFlow/Keras model from a local audio dataset.
- Configure epochs, batch size, learning rate, and test-set proportion from the interface.
- Extract spectrogram features with Librosa and apply audio augmentation.
- Upload a local Keras model or download one from Google Drive.
- Predict a spoken digit from an uploaded WAV file.
- Record a three-second audio sample when local microphone access is available.
- Display prediction confidence and training accuracy/loss curves.
- Inspect TensorFlow/Keras or PyTorch model structure with Plotly and NetworkX visualizations.
- Use GPU memory growth and mixed precision when enabled and supported.

## Project structure

- `streamlit_app.py` — Streamlit interface, navigation, training, prediction, and model visualization.
- `advanced_speech_recognition.py` — feature extraction, augmentation, model construction, training, persistence, and prediction support.
- `config.py` — dataset, model, training, performance, and logging configuration.
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python runtime declaration.

## Requirements

- Python 3.10, as declared in `runtime.txt`
- A directory containing the WAV dataset organized according to the subdirectories configured in `config.py`
- A compatible pretrained model for prediction, unless a model is trained through the application

GPU acceleration is optional. TensorFlow falls back to CPU when no compatible GPU is available.

## Installation

```bash
git clone https://github.com/Jalal-Zerroudi/Streamlit-And-ASP.git
cd Streamlit-And-ASP

python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate
```

Install the declared dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The optional microphone-recording path imports `sounddevice`. Install it separately when that feature is needed:

```bash
pip install sounddevice
```

## Run

```bash
streamlit run streamlit_app.py
```

Streamlit will print the local application URL in the terminal.

## Usage

### Train a model

1. Open **Train Model**.
2. Enter the dataset directory.
3. Configure the training parameters.
4. Start training.
5. Review the accuracy and loss charts.

The dataset directory and its class subdirectories must match the values expected by `config.py`. Audio files are expected to use the configured extension, currently WAV in the model workflow.

### Make a prediction

1. Open **Make Prediction**.
2. Upload a local `.h5` model or download one from Google Drive.
3. Upload a WAV recording or use the optional three-second microphone recorder.
4. Select **Predict Digit** to view the predicted class and confidence.

### Inspect a model

Open **Model Information**, then upload or download a supported model. The application attempts TensorFlow/Keras loading first and then PyTorch loading, and displays the information supported by the detected model type.

## Notes

- Uploaded and downloaded models are stored locally in `models/` while the application is running.
- Recorded or uploaded prediction audio is written to `temp_audio.wav`.
- Training can be computationally intensive; duration depends on dataset size, selected parameters, and available hardware.
