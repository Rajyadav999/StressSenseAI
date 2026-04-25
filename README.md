# StressSenseAI – Mental Stress Detection System

StressSenseAI is a machine learning-powered web application designed to assess and analyze mental stress levels using a structured questionnaire and predictive modeling.

It combines psychological assessment with ML-based classification, interactive visualizations, and report generation to provide meaningful insights into user stress patterns.

---

## 🧠 Key Features

* 18-question stress assessment (psychological, behavioral, cognitive)
* Machine Learning-based stress prediction using trained models
* Voice-enabled interaction (speech-to-text & text-to-speech)
* Interactive visualizations:

  * Gauge chart (stress level)
  * Radar chart (category-wise analysis)
  * Bar chart (response breakdown)
* Personalized stress management recommendations
* Role-based insights via dashboard
* SQLite-based response tracking
* Downloadable PDF reports

---

## ⚙️ Tech Stack

**Backend:**

* Python (Flask-based web app)

**Frontend:**

* HTML (rendered via Flask)
* Browser APIs for speech recognition & TTS

**Machine Learning:**

* Scikit-learn
* Joblib (model serialization)

**Database:**

* SQLite

**Visualization & Reports:**

* Matplotlib
* ReportLab

---

## 🤖 Machine Learning Details

* Model Type: (e.g., Logistic Regression / Random Forest — update this)
* Input: Questionnaire responses
* Output: Stress Level Classification (Low / Medium / High)
* Model stored in: `ml_model/`

---

## 📁 Project Structure

```
.
├── stress_app.py        # Main application server
├── ml_model/            # ML models and training utilities
├── data/                # Runtime data files
└── stress_data.db       # SQLite database (generated at runtime)
```

---

## 🚀 Getting Started

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Environment

**Windows:**

```bash
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install matplotlib numpy pandas reportlab scikit-learn joblib
```

---

## ▶️ Run the Application

```bash
python stress_app.py
```

Open in browser:

```
http://localhost:5050
```

Dashboard:

```
http://localhost:5050/dashboard
```

---

## 📊 Output & Reports

* Real-time stress level prediction
* Visual insights using charts
* Downloadable PDF report for each assessment
* Historical tracking via dashboard

---

## ⚠️ Important Notes

* `stress_data.db` is runtime-generated → do NOT commit
* Use `.gitignore` for:

  * `.venv/`
  * `__pycache__/`
* Large ML models → use Git LFS or provide download instructions
* This system is **not a medical diagnostic tool**

---

## 🎯 Project Goal

To provide an accessible and intelligent system for early stress detection using machine learning and user-friendly interaction.

---

## 📌 Future Improvements

* Deep Learning-based prediction
* Real-time emotion detection
* Doctor recommendation system
* Cloud deployment (AWS / Render)
* User authentication system

---

## 📜 License

Add your preferred license (MIT recommended)
