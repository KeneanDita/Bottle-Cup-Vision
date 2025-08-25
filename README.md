# BottleCupVision

**BottleCupVision** is a simple image classification project built with **TensorFlow/Keras** that identifies whether an image contains a **bottle** or a **cup**. This project includes the full training pipeline, evaluation metrics, and a lightweight **Flask web app** for image prediction.

 Author : [Kenean Dita](https://github.com/keneandita)

## Project Overview

* **Goal:** Build an accurate image classifier to distinguish between bottles and cups using everyday images.

* **Dataset:** Kaggle [Bottles and Cups Dataset](https://www.kaggle.com/datasets/dataclusterlabs/bottles-and-cups-dataset?utm_source=chatgpt.com)

* **Model:** Sequential Convolutional Neural Network (CNN) trained in TensorFlow/Keras.

* **Training Details:**

  * Input image size: 128×128
  * Optimizer: Adam
  * Loss: Binary Crossentropy
  * Epochs: 20
  * Batch size: 32
  * Early stopping & model checkpointing implemented

* **Project Structure:**

```
BottleCupVision/
│
├── data/               # Dataset (train/val/test folders)
├── notebooks/          # Exploration notebook with training and evaluation
│   └── exploration.ipynb
├── src/                # Modular source code for data, model, training
├── saved_models/       # Trained models (best_model.h5, final_model/)
├── results/            # Saved plots (training curves, confusion matrix, predictions)
├── templates/          # Flask HTML templates
├── static/             # Uploaded images
├── app.py              # Flask web app
├── requirements.txt    # Python dependencies
└── README.md
```

## Model Performance

* **Accuracy:** 93.33%
* **Classification Report:**

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Bottle       | 0.93      | 1.00   | 0.97     | 14      |
| Cup          | 0.00      | 0.00   | 0.00     | 1       |
| **Accuracy** | -         | -      | 0.93     | 15      |
| Macro Avg    | 0.47      | 0.50   | 0.48     | 15      |
| Weighted Avg | 0.87      | 0.93   | 0.90     | 15      |

> Note: The dataset is imbalanced, which is why the recall for “cup” is low. Future work could include data augmentation or class balancing techniques.

## 📈 Results

* **Training Curves:** `Results/Accuracy Loss.png`

![Training Curves](Results/Accuracy%20Loss.png)

* **Confusion Matrix:** `results/confusion_matrix.png`

![Confusion Matrix](Results/Confusion%20Mmatrix.png)

* **Sample Predictions:** `results/sample_predictions.png`

![Sample Predictions](Results/Sample%20outputs.png)

## Web App

* **Built with:** Flask
* **Features:**

  * Upload an image via web interface
  * Predict whether it’s a bottle or a cup
  * Display uploaded image and prediction

**Run locally:**
Clone the Repo:

```bash
git clone https://github.com/KeneanDita/Bottle-Cup-Vision
cd .\Bottle-Cup-Vision
```

Create a virtual evnironment and install the dependencies

```bash
python -m venv env
.\env\source\activate  # for windows
source env\source\activate # for linux/mac
pip install -r requirements.txt
```

```bash
python app.py
```

* Access: `http://127.0.0.1:5000/`

## Docker Image

You can also run BottleCupVision via Docker:

[![Docker Image](https://img.shields.io/badge/Docker-Container-blue)](https://hub.docker.com/repository/docker/yourusername/bottlecupvision)

## Future Enhancements(Rare)

* Implement **transfer learning** (e.g., MobileNetV2) to improve accuracy.
* Balance dataset for better recall on minority classes.
* Extend to **object detection** using the XML annotations.
