# Automated Credit Risk Assessment: Predicting Loan Approvals with Deep Learning and Ensemble Methods

## 📌 Overview
This project explores the automated prediction of loan approvals and credit risk using machine learning. It provides a direct performance comparison between a state-of-the-art ensemble method (**HistGradientBoosting**) and a Deep Learning approach (**Lightweight Artificial Neural Network**). By analyzing applicant financial data (such as income, age, and loan amount), the models classify whether a loan application should be approved or denied.

## 🚀 Features
* **Exploratory Data Analysis (EDA):** Visualizations of applicant distributions (income, age, loan amounts) and feature correlations.
* **Data Preprocessing:** Robust scaling and tensor conversions for deep learning compatibility.
* **HistGradientBoosting (HGB):** A fast, tree-based ensemble classifier optimized for tabular data natively supporting missing values.
* **Lightweight ANN:** A custom-built neural network designed to capture complex, non-linear relationships in applicant financial profiles.
* **Comprehensive Evaluation:** Side-by-side performance metrics using Accuracy Scores, Classification Reports, and visual Confusion Matrices.

## 🛠️ Technologies & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Deep Learning:** PyTorch / TensorFlow *(Note: Update this based on the specific framework used for your tensors)*
* **Data Visualization:** `matplotlib`, `seaborn`

## 📊 Visualizations
*Here is a glimpse of the Exploratory Data Analysis and Model Evaluation:*
* **Distributions:** Log-scaled applicant income, loan amounts, and age distributions.
* **Correlations:** Heatmap analysis of numeric variables to identify multicollinearity.
* **Model Performance:** Seaborn-styled confusion matrices to visualize False Positives and False Negatives.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdullahSajid007/Automated-Credit-Risk-Assessment-Predicting-Loan-Approvals-with-Deep-Learning-and-Ensemble-Methods.git
   cd Automated-Credit-Risk-Assessment-Predicting-Loan-Approvals-with-Deep-Learning-and-Ensemble-Methods
````

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

The notebook will output the training progress, accuracy scores, classification reports, and display the plotted charts.
