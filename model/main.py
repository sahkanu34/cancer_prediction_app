import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Load the data
data = pd.read_csv("E:\Streamlit-App-Cancer\data\data.csv")
data.head()

# Data Preprocessing

# Drop unnecessary columns and handle missing values
data = data.drop('Unnamed: 32', axis=1, errors='ignore')
data = data.drop('id', axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preprocessing and model
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the model
# Create model directory if it doesn't exist
import os
os.makedirs('E:/Streamlit-App-Cancer/model', exist_ok=True)

# Explicitly use full path when saving
joblib.dump(pipeline, 'E:/Streamlit-App-Cancer/model/breast_cancer_model.pkl')
