# Legal Assistant Advisor ⚖️🤖

An AI-powered Legal Assistant web application built using Natural Language Processing (NLP) and Machine Learning to provide automated responses to legal queries. This project was developed as a capstone project and demonstrates an end-to-end AI system, from dataset preparation and model training to deployment with a web-based interface.

---

## 📌 Problem Statement

Accessing basic legal information can be difficult for individuals without legal expertise. This project aims to bridge that gap by providing an intelligent assistant capable of understanding common legal queries and responding with relevant information using machine learning techniques.

---

## 🚀 Features

- AI-based legal query understanding using NLP
- Custom-trained machine learning model
- Web-based user interface for interaction
- Python backend for inference and request handling
- Structured dataset for legal questions and intents
- Modular and extensible architecture

---

## 🧠 System Architecture

1. **Dataset Preparation**
   - Legal queries and intents stored in a structured JSON format

2. **Model Training**
   - NLP preprocessing
   - Feature extraction
   - Model training and persistence

3. **Backend**
   - Python-based application to load the trained model
   - Handles user input and returns predictions

4. **Frontend**
   - HTML, CSS, and JavaScript-based UI
   - Sends queries to backend and displays responses

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning / NLP:** Scikit-learn, NLP preprocessing
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask-style architecture)
- **Data Format:** JSON

---

## 📁 Project Structure

```text
Legal-Assistant-Advisor/
├── legal_assistant_model/   # Saved trained ML model
├── train_model.py           # Model training script
├── app.py                   # Backend application
├── dataset.json             # Legal dataset
├── index.html               # Frontend UI
├── script.js                # Frontend logic
├── styles.css               # Styling
├── requirements.txt         # Dependencies
├── venv/                    # Virtual environment
└── README.md
