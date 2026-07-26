# 🕵️ Deepfake Video Detection Using CNN-Based Models

An end-to-end deepfake detection system leveraging Transfer Learning with the MobileNetV2 architecture. This project automates the ingestion of the Celeb-DF-v2 dataset, extracts spatial features from video frames, trains a binary classification model (Real vs. Fake), and deploys the inference pipeline via an interactive Gradio web interface.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Future Enhancements](#future-enhancements)

## 📖 Project Overview
Deepfakes pose a significant threat to digital trust and security. This project aims to build a scalable detection pipeline that can classify videos as real or artificially synthesized. By extracting individual frames from videos and processing them through a Convolutional Neural Network (CNN), the system captures the spatial artifacts and inconsistencies commonly left behind by deepfake generation algorithms.

## 📁 Repository Structure

```text
deepfake-detection-cnn/
├── .gitignore              # Ignores large datasets and environment files
├── requirements.txt        # Python dependencies
├── download_data.py        # Automates dataset acquisition via Kaggle API
├── preprocess.py           # Extracts and resizes frames from raw .mp4 files
├── train.py                # Compiles and trains the MobileNetV2 model
├── app.py                  # Launches the Gradio web interface for testing
└── README.md               # Project documentation
```

## ⚙️ Prerequisites
* **Compute:** A GPU-enabled environment is highly recommended for training (e.g., Google Colab with T4 GPU or local NVIDIA GPU).
* **Python:** Python 3.8+
* **Kaggle Account:** You will need a Kaggle API token (`KGAT_...`) to download the dataset.

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection-cnn.git
cd deepfake-detection-cnn
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Kaggle API:**
Ensure your Kaggle API token is set as an environment variable before attempting to download the dataset.
* On Linux/macOS/Colab:
```bash
export KAGGLE_API_TOKEN="KGAT_YOUR_TOKEN_HERE"
```
* On Windows (Command Prompt):
```cmd
set KAGGLE_API_TOKEN=KGAT_YOUR_TOKEN_HERE
```

## 💻 Usage Guide

### 1. Download the Dataset
Run the download script to automatically fetch and unzip the Celeb-DF-v2 dataset into the `deepfake_dataset/` directory.
```bash
python download_data.py
```

### 2. Preprocess the Video Data
Execute the preprocessing script to recursively search for `.mp4` files, extract an evenly spaced sample of frames, resize them to 224x224 pixels, and organize them into `dataset/real` and `dataset/fake` directories.
*(Note: The default script processes a subset of 50 videos per class for rapid prototyping).*
```bash
python preprocess.py
```

### 3. Train the Model
Run the training script to initialize the MobileNetV2 base, compile the custom classification head, and train the model on your extracted frames. Once complete, it saves the weights to `deepfake_detector_model.h5`.
```bash
python train.py
```

### 4. Launch the Web Interface
Start the Gradio application to test your trained model. This will generate a local URL and a public `.gradio.live` link where you can upload videos and receive real-time detection scores.
```bash
python app.py
```

## 🧠 Model Architecture
This project utilizes **Transfer Learning** to achieve high accuracy without requiring days of training time.
* **Base Model:** `MobileNetV2` (Pre-trained on ImageNet). The base layers are frozen to retain their robust feature-extraction capabilities.
* **Input Shape:** `(224, 224, 3)`
* **Pooling:** `GlobalAveragePooling2D` to reduce spatial dimensions and parameter count.
* **Regularization:** `Dropout (0.5)` applied to mitigate overfitting.
* **Classification Head:** A single `Dense` neuron with a `sigmoid` activation function to output a binary probability (0 = Fake, 1 = Real).
* **Loss Function:** `binary_crossentropy`

## 🔮 Future Enhancements
* **Facial Cropping (MTCNN):** Upgrade the `preprocess.py` pipeline to utilize `facenet-pytorch`. By detecting and cropping out the background, the CNN can focus exclusively on facial artifacts, dramatically improving accuracy.
* **Temporal Sequence Modeling:** Integrate a Long Short-Term Memory (LSTM) network or 3D Convolutional layers after the CNN feature extraction to analyze frame-to-frame inconsistencies (e.g., unnatural blinking patterns).
