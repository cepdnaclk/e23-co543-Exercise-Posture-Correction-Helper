# Exercise Recognition Model

## Overview

This project uses a PyTorch-based machine learning model to recognize exercises from webcam input using MediaPipe and OpenCV. Due to the size of the training dataset and computational requirements, model training was performed on the University Tesla server, while the live application was developed and tested locally in a dedicated Conda environment.

---

## Workflow

### 1. Dataset Preparation

The exercise dataset (CSV files downloaded from Kaggle) was uploaded to the University Tesla server. Training was performed on the server because:

- The dataset is large and computationally intensive.
- GPU resources on the Tesla server significantly reduce training time.
- Local hardware RAM and processing capabilities were insufficient for efficient training.

---

### 2. Model Training

A Python script named `Train.py` was executed on the university server.

The script:
- Reads the CSV dataset.
- Preprocesses the pose landmark data.
- Trains the exercise classification model using **PyTorch**.
- Saves the trained model and preprocessing objects.

Running `Train.py` generated the following files:

```
best_pose_model.pth
label_encoder.pkl
pose_scaler.pkl
live_pose_app.py
```

---

### 3. Downloading the Trained Files

After training completed, the generated files were copied from the Tesla server to the local machine using `scp`.

Example:

```bash
scp username@tesla.ce.pdn.ac.lk:/path/to/files/* .
```

---

### 4. Local Development Environment

To avoid dependency conflicts and keep the project isolated, a new Conda environment was created.

Environment name:

```
pose_env
```

Example:

```bash
conda create -n pose_env python=3.12
conda activate pose_env
```

---

### 5. Installing Dependencies

All required Python packages were installed inside the Conda environment.

```bash
pip install -r requirements.txt
```

The complete dependency list is available in:

```
requirements.txt
```

---

### 6. Live Webcam Application

Several iterations of `live_pose_app.py` were required before obtaining a stable version due to package version incompatibilities between:

- PyTorch
- OpenCV
- MediaPipe
- Other supporting libraries

The current version successfully integrates:

- Webcam input
- MediaPipe pose estimation
- OpenCV visualization
- PyTorch inference

---

## Technologies Used

- Python
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Conda
- Kaggle Dataset

---

## Current Project Status

### Update (19/07/2026)

Current progress:

- ✅ MediaPipe pose detection is functioning correctly.
- ✅ OpenCV webcam integration is working.
- ✅ The trained model loads successfully.
- ✅ Real-time pose landmarks are extracted correctly.

### Known Issue

The application currently does **not automatically identify which exercise is being performed**.

Instead, it remains fixed on the first exercise in the predefined list (currently **Pull Ups**).

This indicates that additional work is required on the prediction pipeline and/or model inference so that the application continuously classifies the current exercise from the detected pose landmarks.

Future work includes:

- Improving real-time exercise classification.
- Updating the inference pipeline to dynamically predict the current exercise.
- Further model training and validation to improve recognition accuracy.
- Testing with additional exercise samples and users.

---

## Project Structure

```
.
├── Train.py
├── live_pose_app.py
├── best_pose_model.pth
├── pose_scaler.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## Running the Application

Activate the Conda environment:

```bash
conda activate pose_env
```

Install dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python live_pose_app.py
```
