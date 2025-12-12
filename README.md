
---

# 🧠 Brain Tumor Detection Using Deep Learning (VGG19 + Flask)

## 📌 Project Overview

This repository contains a complete AI-powered **Brain Tumor Detection System** built using deep learning and deployed as a real-time web application. The system classifies MRI images into **tumor vs. no tumor** using a **transfer-learned VGG19 CNN** and provides fast predictions using a **Flask backend**.

---

## 🧠 Architecture

The project comprises the following core components:

### 🧩 1. Data & Preprocessing

* **Dataset Structure**: MRI images categorized into `yes/` (tumor) and `no/` (no tumor) folders.
* **Preprocessing Steps**:

  * Resize images to uniform dimensions suitable for VGG19 (e.g., 224×224).
  * Normalize pixel values.
  * (Optional) Augmentation: rotations, flips, zooms to increase data variability and improve generalization.

This pipeline ensures high-quality inputs for model training and inference.

---

### 🧠 2. Deep Learning Model: VGG19

The backbone of the detection system is a **VGG19 Convolutional Neural Network** with transfer learning:

* Loaded with pre-trained ImageNet weights.
* Final classification layers removed and replaced with custom dense layers for **binary classification** (tumor vs. non-tumor).
* Fine-tuning strategy:

  * Initially freeze base layers.
  * Gradually unfreeze deeper layers for better task-specific feature learning.

Achieves high classification accuracy (~90%) on test MRI data.

---

### 📈 3. Training Pipeline

Key Training Components:

* **Training/Validation Split**: Partitioned dataset to monitor overfitting and model performance.
* **Optimization**: Standard Adam optimizer with learning rate adjustments and early stopping.
* **Regular Evaluation**: Track training/validation loss and accuracy metrics for reliable convergence.

This ensures a robust and generalizable model.

---

### ▶️ 4. Deployment with Flask Web App

Once the model is trained and exported:

* **Flask Server** hosts a REST API.
* **User uploads MRI image** through a web interface.
* Server:

  1. Receives the image.
  2. Applies same preprocessing as training.
  3. Loads the trained VGG19 model.
  4. Outputs prediction — **Tumor detected** or **No tumor** — in < 2 seconds.

The Flask app makes the AI system accessible without requiring users to run the model locally.

---

## 📌 Workflow Summary

```
Raw MRI Images
     ↓
Preprocessing (Resize, Normalize, Augment)
     ↓
Transfer Learning with VGG19
     ↓
Train → Validate → Evaluate
     ↓
Save Trained Model
     ↓
Flask Web API
     ↓
User uploads MRI → Returns Tumor Prediction
```

---

## 🛠 Technology Stack

| Component        | Technology               |
| ---------------- | ------------------------ |
| Deep Learning    | TensorFlow / Keras       |
| Pretrained CNN   | VGG19                    |
| Image Processing | OpenCV / PIL             |
| Web App & API    | Flask                    |
| Visualization    | Matplotlib / TensorBoard |

---

## 🚀 How to Use

### Clone & Setup

```bash
git clone https://github.com/RISHI9211/Brain-tumor.git
cd Brain-tumor
pip install -r requirements.txt
```

### Run Training (Notebook / Script)

Follow the notebook to preprocess data, train model, fine-tune, and export the weights.

### Launch Flask App

```bash
flask run
```

Visit `http://127.0.0.1:5000` to upload an MRI image and see live tumor predictions.

---

## 🧪 Results

* **Classification Accuracy**: ~90% on test dataset
* **Prediction Time**: ~2 seconds per image
  (Actual performance may vary with dataset and hardware)

---

## 📂 Repository Structure

```
Brain-tumor/
│── brain_tumor_dataset/         # MRI images (yes/no)
│   ├── yes/
│   └── no/
│── Advance DL Project Brain Tumor Image Classification.ipynb  # Training notebook
│── model/                       # Saved trained model
│── app.py / run.py              # Flask app
│── requirements.txt
│── README.md
```

---

## 🧠 Contributing

Feel free to:

* Improve preprocessing & augmentation
* Add more CNN architectures
* Build a React/Vue frontend
* Deploy on cloud (Heroku / AWS / GCP)

---

## 📄 License

This project is open-source and can be freely used & modified.

---

