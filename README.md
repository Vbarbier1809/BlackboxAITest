# CNN Project

This repository contains a Convolutional Neural Network (CNN) implementation for image classification tasks, specifically demonstrated with the MNIST dataset.

## Files

- `cnn_mnist.py`: Basic CNN implementation for MNIST digit recognition
- `data_preprocessing.py`: Utilities for data preprocessing and augmentation
- `model_evaluation.py`: Functions for model evaluation and visualization
- `train_cnn.py`: Advanced training script with callbacks and evaluation
- `requirements.txt`: Python dependencies

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Vbarbier1809/BlackboxAITest.git
   cd BlackboxAITest
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic MNIST Classification
Run the basic CNN model:
```bash
python cnn_mnist.py
```

### Advanced Training
For more advanced training with early stopping and model checkpointing:
```bash
python train_cnn.py
```

## Features

- Convolutional Neural Network architecture
- Data preprocessing and augmentation
- Model evaluation with confusion matrix
- Training history visualization
- Early stopping and model checkpointing

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
