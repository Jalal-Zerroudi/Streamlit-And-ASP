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

### Audio and model compatibility

The current preprocessing in `advanced_speech_recognition.py` reads audio with SoundFile, averages stereo channels to mono, and computes a decibel STFT spectrogram using `n_fft=2048` and `hop_length=512`. With the default `max_pad_len=173`, each sample has shape `(1025, 173, 1)` before adding the batch dimension. Longer spectrograms are truncated; shorter ones are padded with zeros.

- **Sample rate:** the file's sample rate is read but no resampling is performed. Prepare training and prediction files at the same rate. The built-in microphone recorder uses **44,100 Hz**; recordings from it may not match a model trained at another rate.
- **Class order:** training assigns labels by the order of `Config.DATASET['subdirs']`, currently `d0` through `d9`. Prediction displays the output index selected by `argmax`, so a loaded model must use the same digit-to-index mapping.
- **Augmentation:** random noise and time stretching are enabled during dataset loading for training, but disabled by default when extracting prediction features.
- **Confidence:** the displayed confidence is the largest model output value, not a guarantee that the predicted digit is correct.

When importing a pretrained model, keep its sample rate, feature-extraction settings, padding length, and class order alongside the model file. The current H5-saving method saves the model itself, not this preprocessing metadata.

### Inspect a model

Open **Model Information**, then upload or download a supported model. The application attempts TensorFlow/Keras loading first and then PyTorch loading, and displays the information supported by the detected model type.

## Notes

- Uploaded and downloaded models are stored locally in `models/` while the application is running.
- Recorded or uploaded prediction audio is written to `temp_audio.wav`.
- Training can be computationally intensive; duration depends on dataset size, selected parameters, and available hardware.

## Local recording and audio storage

- **Microphone location:** `sounddevice.rec` runs in the Python process, so
  **Record Audio** uses the microphone attached to the machine hosting Streamlit,
  not the visitor's browser microphone. On a remote or headless server, prefer
  **Upload WAV File**; browser microphone permission alone does not enable this recorder.
- **Shared temporary file:** uploads and recordings both overwrite
  `temp_audio.wav` in the process working directory. The prediction button can
  reuse that file whenever it exists, including audio left by an earlier session.
  Upload or record a fresh sample before predicting.
- **Deployment caution:** this filename is not isolated per user. Concurrent
  sessions can overwrite or predict each other's audio; use the current app as
  a single-user local demo, not as an isolated multi-user audio service.
- **Retention:** the application has no automatic cleanup for `temp_audio.wav`.
  After stopping the app, remove that file if the recording should not remain on disk.
