import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load the data
data = pd.read_csv("E:/Streamlit-App-Cancer/data/data.csv")
print(data.head())

# Data Preprocessing
data = data.drop('Unnamed: 32', axis=1, errors='ignore')
data = data.drop('id', axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Handle Imbalanced Data (if necessary)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    # Create pipeline with preprocessing and model
    pipeline = make_pipeline(
        StandardScaler(),
        model
    )
    
    # Hyperparameter Tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-Validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores for {model.__class__.__name__}:", cv_scores)
    print(f"Mean Cross-Validation Accuracy for {model.__class__.__name__}:", np.mean(cv_scores))
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Model evaluation
    print(f"Accuracy for {model.__class__.__name__}:", accuracy_score(y_test, y_pred))
    print(f"\nClassification Report for {model.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return best_model

# Define models and their parameter grids
models = [
    {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'param_grid': {
            'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__solver': ['liblinear']
        }
    },
    {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [None, 10, 20, 30],
            'randomforestclassifier__min_samples_split': [2, 5, 10]
        }
    },
    {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'gradientboostingclassifier__n_estimators': [100, 200, 300],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
            'gradientboostingclassifier__max_depth': [3, 5, 7]
        }
    }
]

# Train and evaluate each model
best_models = {}
for model_config in models:
    model = model_config['model']
    param_grid = model_config['param_grid']
    best_model = train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test)
    best_models[model.__class__.__name__] = best_model

# Save the best models
os.makedirs('E:/Streamlit-App-Cancer/model', exist_ok=True)
for model_name, model in best_models.items():
    joblib.dump(model, f'E:/Streamlit-App-Cancer/model/{model_name}_breast_cancer_model.pkl')
    print(f"Saved {model_name} model to disk.")