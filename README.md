# Real-Time Hand Sign Translator

A real-time sign language recognition application built using Python, OpenCV, MediaPipe, and Scikit-Learn. This project captures video feed from a camera, extracts 21 3D hand landmarks, and classifies hand signs using a Random Forest machine learning model.

---

## Features
* **Hand Landmark Detection:** Uses MediaPipe to track 21 3D spatial points per hand.
* **ML-Based Classification:** Predicts static hand signs using a trained Random Forest model.
* **Dataset Collection Tool:** Built-in script to easily capture custom hand sign data.
* **Real-Time Translation UI:** Live video display with predicted signs overlaid.

---

## Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** Scikit-Learn, NumPy, Pandas

---

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/Hand-Sign-Translator.git]
cd Hand-Sign-Translator