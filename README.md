# 🧠 Attention-Enhanced Deepfake Detection with EfficientNet-B0 and CBAM

A deep learning-based deepfake detection system built with PyTorch and an attention-enhanced EfficientNet-B0 architecture. The system integrates the Convolutional Block Attention Module (CBAM) to improve the model's ability to focus on discriminative channel features and spatial regions associated with synthetic facial artifacts.

The system supports image and video analysis through a user-friendly web interface and provides a complete training, evaluation, inference, and model export pipeline.

## ⚙️ Created By

- 👨‍💻 [TIENTCHEU TAKOU MARUIRCE DONALD](https://github.com/tientcheudonald237/DeepfakeDetector)


---

## 🌟 Features

- **Attention-Enhanced Deep Learning Model**: EfficientNet-B0 enhanced with a Convolutional Block Attention Module (CBAM)
- **Channel and Spatial Attention**: CBAM helps the model emphasize informative feature channels and spatial regions containing potential synthesis artifacts
- **Multi-format Support**: Analyze both images (.jpg, .jpeg, .png) and videos (.mp4, .mov)
- **Web Interface**: Interactive Gradio-based web application for easy testing
- **Real-time Analysis**: Process first frame of videos for quick deepfake detection
- **Training Pipeline**: Complete PyTorch Lightning training infrastructure
- **Model Export**: Support for PyTorch (.pt) and ONNX format exports

## 🧠 Attention-Enhanced Model Architecture

The proposed model is based on EfficientNet-B0 enhanced with the Convolutional Block Attention Module (CBAM).

The architecture follows the pipeline:

Input Image
↓
EfficientNet-B0 Feature Extractor
↓
Channel Attention
↓
Spatial Attention
↓
Global Average Pooling
↓
Dropout
↓
Fully Connected Classifier
↓
Real / Fake Prediction

### Convolutional Block Attention Module (CBAM)

CBAM is composed of two sequential attention mechanisms:

#### 1. Channel Attention

The Channel Attention Module identifies which feature channels are most informative for the classification task. It uses both average pooling and max pooling to generate channel-wise attention weights.

This allows the network to emphasize feature representations that may be relevant to synthetic facial artifacts.

#### 2. Spatial Attention

The Spatial Attention Module determines which spatial regions of the feature map are more informative. It aggregates channel information using average and max pooling, followed by a convolutional operation.

This allows the model to focus on potentially manipulated facial regions such as:

- Eyes
- Mouth
- Skin regions
- Facial boundaries
- Texture inconsistencies

The resulting attention-refined feature representation is then passed to the classification head.


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TRahulsingh/DeepfakeDetector.git
   cd DeepfakeDetector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a pre-trained model** (or train your own):
   - Place your model file as `models/best_model-v3.pt`

### Usage

#### 🖥️ Web Application

Launch the interactive web interface:

```bash
python web-app.py
```

The web app will open in your browser where you can:
- Drag and drop images or videos
- View real-time predictions with confidence scores
- See preview of analyzed content

#### 🔍 Command Line Classification

Classify individual images:

```bash
python classify.py path/to/your/image.jpg
```

#### 🎥 Video Analysis

Process videos from a folder:

```bash
# Place videos in 'videos_to_predict' folder, then run:
python inference/video_inference.py
```

## 📂 Supported Datasets

### 🖼️ Image-based Datasets

#### **140k Real and Fake Faces**
- **Description**: Large collection of real and AI-generated face images
- **Size**: ~140,000 images
- **Source**: StyleGAN-generated faces vs real faces
- **Download**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)
- **Usage**: Perfect for image-based deepfake detection training

### 🔧 Dataset Preparation

#### Option 1: Download Pre-processed Datasets
1. Download your chosen dataset from the links above
2. Extract to the `data/` folder
3. Organize as shown in the training section below

#### Option 2: Use Dataset Preparation Tools
Use our built-in tools to prepare datasets. Edit the source/destination paths inside each script before running:

```bash
# Extract frames from videos (every 15th frame) and split into train/val
# Edit source & dest paths in the script, then run:
python tools/split_video_dataset.py

# Split an existing image dataset into 80/20 train/validation
# Edit source_dataset & destination paths in the script, then run:
python tools/split_train_val.py

# Extract frames from a single video directory
# Edit video_dir & output_dir in the script, then run:
python tools/split_dataset.py
```

### ⚠️ Dataset Usage Notes

- **Ethical Use**: These datasets are for research purposes only
- **Legal Compliance**: Ensure compliance with dataset licenses and terms of use
- **Privacy**: Respect privacy rights of individuals in the datasets
- **Citation**: Properly cite the original dataset papers when publishing research

## 🏋️ Training

### Dataset Structure

Organize your training data in the `data` folder as follows:
```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── fake/
│       ├── fake1.jpg
│       └── fake2.jpg
└── validation/
    ├── real/
    └── fake/
```

### Configuration

Update `config.yaml` with your dataset paths:

```yaml
train_paths:
  - data/train

val_paths:
  - data/validation

lr: 0.0001
batch_size: 4
num_epochs: 10
use_cbam: False
DEBUG_MODE: False
```

### Start Training

```bash
python main_trainer.py
```

The training will:
- Use PyTorch Lightning for efficient training
- Save best model based on validation loss
- Log metrics to TensorBoard
- Apply early stopping to prevent overfitting

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

## 📁 Project Structure

```
├── web-app.py                    # Main web application
├── main_trainer.py               # Primary training script
├── classify.py                   # Image classification utility
├── realeval.py                   # Real-world evaluation script
├── config.yaml                   # Training configuration
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── ARCHITECTURE.md               # System architecture & design
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── data/                         # Dataset storage (not tracked by git)
│   ├── train/                    # Training data
│   └── validation/               # Validation data
├── datasets/
│   └── hybrid_loader.py          # Custom dataset loader
├── lightning_modules/
│   └── detector.py               # PyTorch Lightning module
├── models/
│   └── best_model-v3.pt          # Trained model weights
├── tools/                        # Dataset preparation utilities
│   ├── export_to_pt.py           # .ckpt → .pt model converter
│   ├── split_dataset.py          # Video frame extractor
│   ├── split_train_val.py        # 80/20 train/val splitter
│   └── split_video_dataset.py    # Video-aware dataset splitter
└── inference/
    ├── export_onnx.py            # ONNX export
    └── video_inference.py        # Multi-frame video inference
```
## 🛠️ Model Architecture

### Backbone

- **Base Model**: EfficientNet-B0
- **Initialization**: ImageNet-pretrained weights
- **Input Size**: 224 × 224 RGB images
- **Feature Dimension**: 1280 channels

### Attention Mechanism

- **Attention Module**: Convolutional Block Attention Module (CBAM)
- **Channel Attention**: Applied to the extracted feature representation
- **Spatial Attention**: Applied after channel refinement
- **Reduction Ratio**: 16
- **Spatial Kernel Size**: 7 × 7

### Classification Head

- **Global Average Pooling**
- **Dropout**: 0.4
- **Fully Connected Layer**: 1280 → 2 classes
- **Output**: Binary classification (Real / Fake)

### Complete Architecture

Input Image (224 × 224 × 3)
↓
EfficientNet-B0 Feature Extractor
↓
1280-Channel Feature Map
↓
CBAM
├── Channel Attention
└── Spatial Attention
↓
Global Average Pooling
↓
Dropout (0.4)
↓
Fully Connected Layer
↓
Real / Fake Prediction


### Model Complexity

The attention-enhanced model contains approximately **4.22 million trainable parameters** and requires approximately **385 MB of multiply-add operations** for a single forward pass at the configured input resolution.

The CBAM module introduces a relatively small computational overhead compared with the complete EfficientNet-B0 backbone while providing additional channel and spatial feature refinement.

## 📈 Results

The CBAM-enhanced EfficientNet-B0 achieved strong validation performance on the 140K Real and Fake Faces dataset.

The best validation performance was obtained around the fourth training epoch, with:

- **Validation Accuracy**: approximately 99.51%
- **Validation Precision**: approximately 99.32%
- **Validation Recall**: approximately 99.70%
- **Validation F1-score**: approximately 99.51%
- **Validation AUROC**: approximately 99.99%

The confusion matrix showed:

- **9,968 correctly classified real images**
- **9,927 correctly classified fake images**
- **32 real images incorrectly classified as fake**
- **73 fake images incorrectly classified as real**

These results demonstrate the strong discriminative capability of the attention-enhanced EfficientNet-B0 architecture on the evaluated dataset.

## 📊 Performance

- **Inference Speed**: Real-time on GPU, ~200ms per image on CPU
- **Input Support**: Images (.jpg, .png) and videos (.mp4, .mov)
- **Video Analysis**: 10-frame uniform sampling with probability averaging
- **Robustness**: Tested with Gaussian blur and JPEG compression noise simulation (`realeval.py`)

> **Note**: Accuracy metrics depend on your training dataset. Monitor `val_loss` and `val_acc` via TensorBoard during training.

## 🔧 Advanced Usage

### Export to ONNX

Convert PyTorch model to ONNX format:

```bash
python inference/export_onnx.py
```

### Batch Evaluation

Evaluate a folder of real-world samples with optional noise simulation:

```bash
# Place test images/videos in realworld_samples/ folder, then run:
python realeval.py
```

### Export Checkpoint to PyTorch

Convert a Lightning `.ckpt` to a standalone `.pt` file:

```bash
# Edit ckpt_path and pt_output in the script, then run:
python tools/export_to_pt.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch Lightning for training infrastructure
- Gradio for web interface framework
- The research community for deepfake detection advances

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ **Star this repository if you found it helpful!**
