# Disease Prediction Using Machine Learning

## Project Overview
This project uses Machine Learning algorithms to classify breast cancer tumors as **Malignant (Cancerous)** or **Benign (Non-cancerous)** using medical diagnostic data.

The project is developed as part of the **CodeAlpha Machine Learning Internship**.

## Dataset
- **Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** Scikit-learn built-in dataset
- **Total Samples:** 569
- **Total Features:** 30
- **Malignant Cases:** 212
- **Benign Cases:** 357
- **Missing Values:** None

## Machine Learning Algorithms
The following algorithms are implemented and compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. XGBoost Classifier

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## Project Workflow
1. Load the breast cancer dataset.
2. Explore the data and visualize class distribution.
3. Split the dataset into training and testing sets.
4. Standardize features where required.
5. Train four Machine Learning models.
6. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
7. Compare model performance.
8. Save the best-performing model using Joblib.

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| Support Vector Machine (SVM) | 98.25% |
| Random Forest | 95.61% |
| XGBoost | 94.74% |

Logistic Regression was selected as the best model based on the macro F1-score. It tied with SVM on the reported metrics.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_Disease_Prediction.git
```

### 2. Open the Project Folder
```bash
cd CodeAlpha_Disease_Prediction
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the Project
```bash
python main.py
```

## Output
The project displays:
- Dataset analysis and visualizations
- Model evaluation reports
- Confusion matrices
- Model performance comparison
- Accuracy comparison graph

The best model is saved as `best_disease_prediction_model.pkl`.

## Disclaimer
This project is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

## Author
Machine Learning Intern — CodeAlpha
