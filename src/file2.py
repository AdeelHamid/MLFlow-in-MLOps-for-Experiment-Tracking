import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='adeel.hamid50',
             repo_name='MLFlow-in-MLOps-for-Experiment-Tracking',
             mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/adeel.hamid50/MLFlow-in-MLOps-for-Experiment-Tracking.mlflow')

# Load the wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

# Define the parameters for random forest
rf_params = {
    "n_estimators": 8,
    "max_depth": 5,
    "min_samples_split": 2,
    "random_state": 42
}

# Set Experiment Name
mlflow.set_experiment("MLOps-Experiment1")

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(**rf_params)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", float(accuracy))   # ✅ ensure scalar float

    # Log parameters
    mlflow.log_params(rf_params)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine_data.target_names,
                yticklabels=wine_data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save and log as artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ✅ Check before logging the script itself
    if os.path.isfile(__file__):
        mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({
        "Author": "Adeel",
        "Project": "Wine Classification"
    })

    # Log the model
    mlflow.sklearn.log_model(rf, "model")

    print("Model Accuracy:", accuracy)
