
# ============================================================
# CODEALPHA PROJECT: DISEASE PREDICTION
# Dataset: Breast Cancer Wisconsin (built into scikit-learn)
# Models: Logistic Regression, SVM, Random Forest, XGBoost
# ============================================================

# Step 1: Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import the dataset
from sklearn.datasets import load_breast_cancer

# Import the function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import preprocessing tools
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import the four machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

# Load the built-in Breast Cancer dataset
data = load_breast_cancer()

# X contains the 30 tumor measurement features
X = pd.DataFrame(data.data, columns=data.feature_names)

# y contains the target labels
# 0 = Malignant, 1 = Benign
y = pd.Series(data.target, name="target")

# Define readable class names
class_names = ["Malignant", "Benign"]

# Display basic dataset information
print("\n" + "=" * 60)
print("DISEASE PREDICTION PROJECT")
print("=" * 60)

print("\nDataset Shape:", X.shape)
print("Total Records:", X.shape[0])
print("Total Features:", X.shape[1])

print("\nFirst 5 Records:")
print(X.head())

print("\nTarget Classes:")
print(data.target_names)

# Display class counts using readable labels
print("\nClass Distribution:")
print(y.map({0: "Malignant", 1: "Benign"}).value_counts())

# Check for missing values in the input features
print("\nTotal Missing Values:", X.isnull().sum().sum())


# ============================================================
# STEP 3: VISUALIZE THE DATA
# ============================================================

# Convert target numbers into readable class names
class_labels = y.map({0: "Malignant", 1: "Benign"})

# Plot the number of samples in each class
plt.figure(figsize=(7, 5))
sns.countplot(x=class_labels, order=class_names)

plt.title("Distribution of Breast Cancer Classes")
plt.xlabel("Cancer Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Plot the distribution of mean radius measurements
plt.figure(figsize=(8, 5))
sns.histplot(data=X, x="mean radius", bins=25, kde=True)

plt.title("Distribution of Mean Radius")
plt.xlabel("Mean Radius")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 4: SPLIT THE DATASET
# ============================================================

# Use 80% of records for training and 20% for testing
# Stratification preserves the class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


# ============================================================
# STEP 5: DEFINE THE FOUR MACHINE LEARNING MODELS
# ============================================================

# Scaling is important for Logistic Regression and SVM.
# Pipeline ensures the scaler is fitted using training data only.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    # Tree-based models generally do not require feature scaling
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# STEP 6: TRAIN AND EVALUATE EACH MODEL
# ============================================================

# Store each model's evaluation results
results = []

# Store the trained models and their predictions
trained_models = {}
model_predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict classes for previously unseen test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Save metrics for the comparison table
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "F1-Score (Macro)": f1
    })

    # Keep the trained model and predictions for later analysis
    trained_models[model_name] = model
    model_predictions[model_name] = y_pred

    # Display the individual model's scores
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1-Score (Macro):  {f1:.4f}")

    # Display the detailed performance for both classes
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    ))

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Display the confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 7: COMPARE ALL FOUR MODELS
# ============================================================

# Convert evaluation results into a DataFrame
results_df = pd.DataFrame(results)

# Sort models by macro F1-score, from highest to lowest
results_df = results_df.sort_values(
    by="F1-Score (Macro)",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

# Display the scores in a readable format
print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Precision (Macro)": "{:.4f}".format,
        "Recall (Macro)": "{:.4f}".format,
        "F1-Score (Macro)": "{:.4f}".format
    }
))


# ============================================================
# STEP 8: PLOT MODEL ACCURACY COMPARISON
# ============================================================

# Plot the accuracy of all four models
plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Accuracy Comparison of Disease Prediction Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 9: SAVE THE BEST MODEL
# ============================================================

# Select the model with the highest macro F1-score
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# Save the model, including any preprocessing pipeline
joblib.dump(best_model, "best_disease_prediction_model.pkl")

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("Best Model by Macro F1-Score:", best_model_name)
print("Macro F1-Score:", f"{results_df.iloc[0]['F1-Score (Macro)']:.4f}")
print("Saved model: best_disease_prediction_model.pkl")
print("\nAll four models have been trained and evaluated.")
