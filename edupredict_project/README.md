# EduPredict Studio – Student Performance Prediction Using Artificial Neural Networks

A complete, end-to-end Deep Learning & Educational Analytics Project developed for the **MBA Data Science & Artificial Intelligence** curriculum. It includes both a **Zero-Setup Single-File Browser Studio (`edupredict_ann_dashboard.html`)** powered by TensorFlow.js, Chart.js, and PapaParse, and an optional **Python / Streamlit companion app (`app.py`)**.

---

## 🌟 Key Features

1. **Synthetic Student Dataset Generator (13 Academic Features)**:
   - Automatically generates realistic student cohorts of **100, 500, 1,000 (default), 2,000, or 5,000 records**.
   - Features: `Student_ID`, `Age` (16–25), `Gender`, `Attendance` (40–100%), `Study_Hours` (0–10 hrs/day), `Previous_Score` (0–100), `Assignment_Score` (0–100), `Participation` (0–100), `Sleep_Hours` (3–10 hrs), `Internet_Access`, `Extracurricular`, `Final_Score` (0–100), and binary `Performance` (`1 = Pass`, `0 = Fail`).
   - Configurable **Passing Score Cutoff** (default `40`) and reproducible PRNG seed.

2. **Custom CSV Dataset Ingestion & Leakage-Free Preprocessing**:
   - Drag & drop or browse custom `.csv` files parsed via **PapaParse**.
   - Automatic detection of numerical vs. categorical columns, missing values, and duplicate rows.
   - **Zero Data Leakage**: Automatically excludes `Student_ID`, `Final_Score`, and `Performance` from model inputs. Fits mean/mode imputation, Z-score standardization $((x - \mu_{\text{train}})/\sigma_{\text{train}})$, and one-hot encoding strictly on the training split.

3. **Genuine Feedforward Artificial Neural Network (TensorFlow.js)**:
   - **Input Layer**: Dynamically sized to the standardized numerical + one-hot encoded categorical features.
   - **Hidden Layer 1**: Configurable neurons (default `16`) with **ReLU** activation $\text{ReLU}(z) = \max(0, z)$.
   - **Hidden Layer 2 (Optional)**: Configurable second ReLU hidden layer (`8`, `12`, or `16` units).
   - **Output Layer**: `1` neuron with **Sigmoid** activation $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputting $P(\text{Pass} \mid \mathbf{x})$.
   - **Optimization**: Trained via backpropagation using the **Adam** optimizer (`learningRate = 0.001`) minimizing **Binary Cross-Entropy Loss**.

4. **Interactive 2D Decision Surface & 18 Live Visualizations**:
   - **2D ANN Decision Surface Canvas**: Visualizes the neural network's probability contour across any two selected academic features with an interactive classification threshold contour ($P = \tau$).
   - **10 Exploratory Dataset Charts**: Pass/Fail balance, Attendance, Study Hours, Previous Score, Assignment Score, Final Score, and Pass/Fail breakdowns by Gender, Attendance Tier, Study-Hour Tier, and Extracurriculars.
   - **8 ANN Diagnostic Charts**: Training Loss, Validation Loss, Training Accuracy, Validation Accuracy, Confusion Matrix, ROC Curve & AUC, Actual vs. Predicted counts, and Predicted Pass Probability distribution, plus post-hoc Permutation Feature Sensitivity.

5. **Individual Student Outcome Simulator & 5 Export Formats**:
   - Real-time Pass/Fail prediction for individual students using training-set preprocessing parameters, plus CSV/TXT/JSON exports for datasets, held-out test predictions, individual student predictions, and evaluation reports.

---

## 🚀 How to Run

### Method 1: One-Click Browser Studio (Recommended — Zero Setup)
Simply open **`edupredict_ann_dashboard.html`** in any modern web browser (Chrome, Edge, Firefox, Safari).
- Runs 100% in the browser using **TensorFlow.js**, **Chart.js**, **PapaParse**, and **Tailwind CSS** via CDN (internet connection required unless CDN scripts are cached locally).
- No Python installation, backend server, or terminal commands required.

### Method 2: Python / Streamlit Companion Version
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
