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

## Tests

The configuration-isolation regression test uses Python's standard-library `unittest` runner and does not require a dataset or trained model:

```bash
python -m unittest discover -s tests -v
```

It verifies that dataset paths, class lists, and training overrides made for one model configuration do not leak into another session or mutate the shared defaults.

## Usage

### Prepare the dataset

The loader reads audio files directly from the ten class directories configured
in `Config.DATASET['subdirs']`. With the default configuration, use this layout:

```text
dataset/
├── d0/
│   ├── sample-001.wav
│   └── sample-002.wav
├── d1/
│   └── sample-001.wav
├── ...
└── d9/
    └── sample-001.wav
```

Point **Dataset Path** at the directory containing `d0` through `d9`, not at an
individual digit directory. Files nested below those class directories are not
scanned. The current loader matches the configured `.wav` suffix exactly, so use
lowercase `.wav` filenames rather than `.WAV`.

Training uses a stratified train/test split. Include examples for every digit and
enough files per class for the selected test proportion to place samples from each
class in both splits. If a class directory is missing or empty, the application
prints a warning; training stops when no usable audio can be loaded.

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

Google Drive downloads accept either a standard share link or the raw file ID.
The app validates the reference and confirms that a non-empty model file was
downloaded before reporting success; a failed download cannot silently reuse an
older model from the same session.

## Notes

- Each model instance receives an isolated copy of the mutable configuration, so dataset paths and training overrides do not leak between Streamlit sessions.
- Uploaded and downloaded models are stored in a unique temporary directory for the current Streamlit session, preventing concurrent sessions from overwriting one another. These directories are not removed automatically; clear the operating system's temporary files when model artifacts should not remain on disk.
- Recorded or uploaded prediction audio is written to a unique temporary WAV file for the current Streamlit session.
- Training can be computationally intensive; duration depends on dataset size, selected parameters, and available hardware.

## Local recording and audio storage

- **Microphone location:** `sounddevice.rec` runs in the Python process, so
  **Record Audio** uses the microphone attached to the machine hosting Streamlit,
  not the visitor's browser microphone. On a remote or headless server, prefer
  **Upload WAV File**; browser microphone permission alone does not enable this recorder.
- **Session isolation:** uploads and recordings use a random filename in the
  operating system's temporary directory. Concurrent Streamlit sessions no
  longer overwrite or predict each other's audio.
- **Session reuse:** the prediction button can reuse the current session's file
  whenever it exists. Upload or record a fresh sample before predicting a new digit.
- **Retention:** the application does not automatically remove the per-session
  WAV file. Clear the operating system's temporary files when recordings should
  not remain on disk.

