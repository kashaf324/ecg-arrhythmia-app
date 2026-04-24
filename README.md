# 🫀 ECG Arrhythmia Classification System

An AI-powered web application for detecting cardiac arrhythmias using machine learning. This project uses ECG feature data to classify heartbeats into normal and abnormal categories with real-time predictions via an interactive Streamlit dashboard.

---

## Live Demo
(Add your Streamlit link here after deployment)

---

## Problem Statement

Cardiac arrhythmias can indicate serious heart conditions. Early detection using ECG data is critical but often requires expert analysis.

This project aims to:
- Automate arrhythmia detection
- Assist in early diagnosis
- Provide an interactive tool for ECG analysis

---

## Dataset

This project uses ECG data from the **MIT-BIH Arrhythmia Dataset (PhysioNet)**.

### Key characteristics:
- Multi-class classification problem
- Classes:
  - **N** → Normal
  - **SVEB** → Supraventricular ectopic beat
  - **VEB** → Ventricular ectopic beat
  - **F** → Fusion beat
  - **Q** → Unknown beat
- 34 engineered ECG features (2 leads)

---

## Features

✔ Upload ECG dataset (CSV)  
✔ Real-time arrhythmia prediction  
✔ Detection of abnormal heartbeats  
✔ Interactive dashboard with metrics  
✔ Visualization of class distribution  

---

## Project Structure
ecg-arrhythmia-app/
│── app.py # Streamlit web app
│── train.py # Model training script
│── model.pkl # Trained ML model
│── scaler.pkl # Feature scaler
│── label_encoder.pkl # Label encoder
│── requirements.txt # Dependencies
│── README.md


---

## Machine Learning Model

- Model: **Random Forest Classifier**
- Evaluation Metric: **Macro F1-score**
- Reason:
  - Handles tabular ECG features effectively
  - Robust to noise and imbalance

---

## Results

- Achieved strong performance on major arrhythmia classes
- High accuracy on normal and ventricular beats
- Identified class imbalance as a key challenge

---

## How to Run Locally

### 1. Clone repository
```bash
git clone https://github.com/your-username/ecg-arrhythmia-app.git
cd ecg-arrhythmia-app

### 2. Install Dependencies
python -m pip install -r requirements.txt

### 3. Run the app
python -m streamlit run app.py

## Input format
Upload a CSV file with ECG features similar to training data.

## Limitations
Works only with feature-based ECG datasets
Performance affected by class imbalance
Not intended for clinical use

Author 
Kashaf Raheem
AI / Software Engineer

