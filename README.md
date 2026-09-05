# Throat-Cancer-Detection-Using-Speech-Signals-and-Machine-Learning
AI-powered throat cancer detection system using speech signals, MFCC feature extraction, and deep learning models such as CNN and Bi-LSTM to classify voice samples as Normal or Potential Throat Cancer.
# Detecting Throat Cancer from Speech Signals Using Machine Learning

## 📌 About the Project

This project focuses on detecting potential throat cancer from **speech signals using Machine Learning and Deep Learning techniques**.

The system analyzes speech samples by performing audio preprocessing and extracting relevant acoustic features such as **Mel-Frequency Cepstral Coefficients (MFCCs)** and mel-spectrograms. These features are then used with machine learning/deep learning models to classify speech samples as **Normal** or **Potential Throat Cancer**.

The objective is to explore a **non-invasive and accessible approach** for early screening of potential throat cancer using speech analysis.

> ⚠️ **Disclaimer:** This project is developed for academic and research purposes. It is not intended to replace professional medical diagnosis.

---

## 🎯 Objectives

* Analyze speech signals for potential indicators of throat cancer.
* Preprocess speech recordings to improve input quality.
* Extract relevant acoustic features from speech.
* Use MFCC and other speech representations for analysis.
* Apply machine learning and deep learning techniques for classification.
* Provide a simple interface for submitting speech samples and viewing results.

---

## 🔍 Problem Statement

Traditional throat cancer diagnosis may involve invasive and costly procedures such as laryngoscopy and biopsy. Voice changes can also occur due to various benign disorders, making early differentiation challenging.

This project explores the use of **speech signals and machine learning** as a non-invasive approach for identifying patterns that may be associated with throat cancer.

---

## 💡 Proposed Solution

The proposed system follows a speech-processing and machine-learning pipeline:

```text
Speech Input
     ↓
Audio Preprocessing
     ↓
Feature Extraction
     ↓
MFCC / Mel-Spectrogram
     ↓
Machine Learning / Deep Learning Model
     ↓
Classification
     ↓
Result
```

The system accepts speech samples, preprocesses the audio, extracts acoustic features, and uses classification models to generate a prediction.

---

## ✨ Features

* 🎙️ Speech recording or upload
* 🔊 Audio preprocessing
* 🧹 Noise reduction and normalization
* 📊 MFCC feature extraction
* 📈 Mel-spectrogram analysis
* 🧠 Machine learning/deep learning classification
* 📋 Prediction result generation
* 👤 User/patient module
* 👨‍💼 Admin module
* 🗄️ Database support

The project specification includes speech input, preprocessing, feature extraction, ML model integration, and classification into **Normal** or **Potential Throat Cancer**.

---

## 🧠 Machine Learning Approach

### 1. Audio Preprocessing

The input speech signal is prepared before feature extraction.

Preprocessing includes:

* Noise reduction
* Normalization
* Audio segmentation
* Improving the quality of the input signal

### 2. Feature Extraction

The system extracts acoustic features from the speech signal.

The main feature used is:

**MFCC — Mel-Frequency Cepstral Coefficients**

Other speech representations include:

* Mel-spectrograms
* Jitter
* Shimmer

These features provide information about the characteristics of the speech signal.

### 3. Classification

The proposed system considers deep learning architectures including:

* CNN
* RNN
* Bi-LSTM

CNN can be used for learning patterns from speech feature representations, while recurrent architectures can capture temporal patterns in speech.

---

## 🔄 System Workflow

```text
                  ┌───────────────────┐
                  │   User / Patient  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   Speech Input    │
                  │  Record / Upload  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Audio Preprocessing│
                  │ Noise Reduction   │
                  │ Normalization     │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Feature Extraction│
                  │ MFCC / Mel-Spec.  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ ML / DL Model     │
                  │ CNN / RNN /       │
                  │ Bi-LSTM           │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   Classification  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   Result Display  │
                  └───────────────────┘
```

---

## 🛠️ Technologies Used

| Category                | Technology                       |
| ----------------------- | -------------------------------- |
| Programming Language    | Python                           |
| Machine Learning        | Machine Learning / Deep Learning |
| Deep Learning           | CNN, RNN, Bi-LSTM                |
| Audio Processing        | Librosa                          |
| Feature Extraction      | MFCC, Mel-Spectrogram            |
| Web Framework           | Flask                            |
| Frontend                | HTML, CSS, JavaScript            |
| Database                | MySQL                            |
| Development Environment | Jupyter Notebook / VS Code       |
| Version Control         | Git / GitHub                     |

The project report specifies Python, Flask, HTML/CSS/JavaScript and MySQL as the software technologies.

---

# 📂 Project Structure

```text
throat-cancer-speech-detection/
│
├── dataset/
│   ├── normal/
│   └── cancer/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── prediction.py
│
├── models/
│   └── trained_model.h5
│
├── templates/
│   ├── index.html
│   ├── login.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── screenshots/
│   ├── home.png
│   ├── speech-input.png
│   └── result.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Important:** Change this structure to match your actual project files before publishing. Do not add files or folders that don't exist in your repository.

---

# ⚙️ Installation

## Prerequisites

Make sure you have the following installed:

* Python 3.x
* pip
* Git
* MySQL
* VS Code or another Python IDE

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/throat-cancer-speech-detection.git
```

Navigate to the project directory:

```bash
cd throat-cancer-speech-detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

Example dependencies used for the project may include:

```text
numpy
pandas
librosa
scikit-learn
matplotlib
seaborn
flask
mysql-connector-python
```

Add the exact versions used by your project to `requirements.txt`.

---

# 🗄️ Database Configuration

The proposed application uses **MySQL** for storing relevant application data.

Create the database:

```sql
CREATE DATABASE throat_cancer_detection;
```

Configure the database connection in the application:

```python
DB_HOST = "localhost"
DB_USER = "your_username"
DB_PASSWORD = "your_password"
DB_NAME = "throat_cancer_detection"
```

### 🔐 Security

Never upload actual passwords, API keys, or other credentials to GitHub.

Use environment variables or a `.env` file for sensitive configuration.

---

# ▶️ Running the Application

After installing the dependencies, run:

```bash
python app.py
```

Open the local Flask URL displayed in your terminal.

Example:

```text
http://127.0.0.1:5000/
```

---

# 📊 Expected Output

After submitting a speech sample, the system performs the processing and classification steps.

Example output:

```text
Speech Analysis Result
----------------------

Prediction:
Normal
```

or:

```text
Speech Analysis Result
----------------------

Prediction:
Potential Throat Cancer
```

The project specification defines these two classification outcomes.

---

# 🖼️ Screenshots

Add screenshots of your **actual working application** to the `screenshots/` folder.

## 🏠 Home Page

```markdown
![Home Page](screenshots/home.png)
```

## 🎙️ Speech Input

```markdown
![Speech Input](screenshots/speech-input.png)
```

## 📊 Feature Extraction

```markdown
![Feature Extraction](screenshots/features.png)
```

## 🧠 Prediction Result

```markdown
![Prediction Result](screenshots/result.png)
```

## 👨‍💼 Admin Dashboard

```markdown
![Admin Dashboard](screenshots/admin-dashboard.png)
```

Your report describes separate user/patient and admin functionality, including speech submission, results, user management, data management, and monitoring.

---

# 📈 Future Enhancements

Potential future improvements include:

* Larger and more diverse speech datasets
* Improved model generalization
* Real-time speech analysis
* Transfer learning
* Explainable AI
* Integration with electronic health records
* Multimodal medical data analysis
* Improved clinical workflow integration
* Greater reproducibility through open-source datasets and code

These enhancements are consistent with the future directions described in the project report.

---

# ⚠️ Limitations

* Speech characteristics vary between individuals.
* Dataset size and diversity can affect model performance.
* Small datasets can increase the risk of overfitting.
* Different recording environments can affect audio quality.
* The model requires proper validation before real-world medical use.
* The system should not be used as a standalone medical diagnostic tool.

The literature review also identifies limited datasets, inconsistent methodologies, and reproducibility challenges as important limitations in speech-based throat-cancer research.

---

# 👥 Contributors

### Mounika Munagala

B.Tech – Information Technology

### Thrishma Rapolu

B.Tech – Information Technology

### Likhitha Talagana

B.Tech – Information Technology

---

# 🎓 Academic Project

**Project:** Detecting Throat Cancer from Speech Signals Using Machine Learning

**Degree:** Bachelor of Technology – Information Technology

**Institution:** Sri Devi Women's Engineering College

**Academic Year:** 2025–2026

---

# 📜 Disclaimer

This project is developed for **academic and research purposes only**.

The prediction generated by this system should not be considered a medical diagnosis or medical advice. Any suspected health condition should be evaluated by a qualified healthcare professional using appropriate clinical diagnostic procedures.

---

# ⭐ Support

If you found this project useful for learning about **Python, speech processing, machine learning, or deep learning**, consider giving the repository a ⭐ on GitHu
