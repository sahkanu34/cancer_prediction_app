import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.pkl")

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
        return None

def main():
    st.set_page_config(page_title='Breast Cancer Diagnosis', layout='centered', initial_sidebar_state='auto')
    
    st.image("https://raw.githubusercontent.com/sahkanu34/cancer_prediction_app/main/app/img.jpeg", 
             use_column_width=True)
    
    st.title('Breast Cancer Diagnosis Predictor')
    
    # Sidebar navigation
    app_mode = st.sidebar.selectbox(
        'Choose Module', 
        ['Prediction', 'Model Details', 'Data Insights'],
        key='main_navigation'
    )
    
    # Load model and data
    model = load_model()
    data = load_dataset()
    
    if model is None or data is None:
        st.error("Unable to load model or dataset. Please check your files.")
        return
    
    # Routing
    if app_mode == 'Prediction':
        prediction_page(model, data)
    elif app_mode == 'Model Details':
        model_info_page(model)
    else:
        dataset_exploration_page(data)

def prediction_page(model, data):
    st.header('Tumor Diagnosis Prediction')
    
    # Prepare feature names
    feature_names = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    
    # Dynamic feature input
    st.sidebar.header('Diagnostic Parameters')
    input_features = {}
    
    for feature in feature_names:
        min_val, max_val = data[feature].min(), data[feature].max()
        mean_val = data[feature].mean()
        
        input_features[feature] = st.sidebar.slider(
            feature.replace('_', ' ').title(), 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(mean_val), 
            step=(max_val - min_val) / 100
        )
    
    # Prediction
    if st.sidebar.button('Analyze Tumor'):
        input_data = pd.DataFrame([input_features], columns=feature_names)
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Result visualization
        if prediction[0] == 1:
            st.error('🚨 Malignant Tumor Detected')
            st.write(f'Malignancy Probability: {prediction_proba[0][1]:.2%}')
        else:
            st.success('✅ Benign Tumor Detected')
            st.write(f'Benignancy Probability: {prediction_proba[0][0]:.2%}')
        
        # Probability bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(['Benign', 'Malignant'], prediction_proba[0])
        plt.title('Tumor Classification Probabilities')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        st.pyplot(plt)

def model_info_page(model):
    st.header('Model Diagnostic Overview')
    
    st.markdown("""
    ### Breast Cancer Classification Model
    - **Algorithm:** Logistic Regression
    - **Preprocessing:** Standard Feature Scaling
    - **Purpose:** Binary Tumor Classification
    """)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "96%")
    with col2:
        st.metric("Precision", "94%")

def dataset_exploration_page(data):
    st.header('Advanced Data Insights')
    
    # Visualization selection
    viz_option = st.selectbox(
        'Choose Visualization', 
        [
            'Overview', 
            'Diagnosis Distribution', 
            'Feature Distributions', 
            'Correlation Heatmap',
            'Feature Importance',
            'Box Plots by Diagnosis',
            'Violin Plots'
        ],
        key='visualization_select'
    )
    
    # Visualization routing
    visualization_map = {
        'Overview': dataset_overview,
        'Diagnosis Distribution': diagnosis_distribution,
        'Feature Distributions': feature_distributions,
        'Correlation Heatmap': correlation_heatmap,
        'Feature Importance': feature_importance,
        'Box Plots by Diagnosis': box_plots,
        'Violin Plots': violin_plots
    }
    
    visualization_map[viz_option](data)

def dataset_overview(data):
    st.subheader('Dataset Statistics')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Samples", len(data))
        st.metric("Malignant Cases", len(data[data['diagnosis'] == 'M']))
    
    with col2:
        st.metric("Features", len(data.columns) - 3)
        st.metric("Benign Cases", len(data[data['diagnosis'] == 'B']))

def diagnosis_distribution(data):
    st.subheader('Tumor Diagnosis Distribution')
    
    diagnosis_counts = data['diagnosis'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.pie(diagnosis_counts, labels=['Benign', 'Malignant'], autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Diagnosis Proportion')
    
    diagnosis_counts.plot(kind='bar', ax=ax2, color=['lightgreen', 'lightcoral'])
    ax2.set_title('Diagnosis Counts')
    ax2.set_xlabel('Diagnosis')
    ax2.set_ylabel('Number of Cases')
    
    st.pyplot(fig)

def feature_distributions(data):
    st.subheader('Feature Distributions')
    
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox(
        'Choose Feature to Visualize', 
        features_to_plot,
        key='feature_distribution_select'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    data[selected_feature].hist(ax=ax1, bins=30)
    ax1.set_title(f'Histogram of {selected_feature}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    data[selected_feature].plot(kind='density', ax=ax2)
    ax2.set_title(f'Density Plot of {selected_feature}')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    
    st.pyplot(fig)

def correlation_heatmap(data):
    st.subheader('Feature Correlation Heatmap')
    
    correlation_matrix = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Diagnostic Features')
    st.pyplot(plt)

def feature_importance(data):
    st.subheader('Feature Importance Analysis')
    
    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(data['diagnosis'])
    
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score', y='feature', data=feature_scores.head(5), 
                palette='viridis')
    plt.title('Top 5 Most Important Features')
    plt.xlabel('F-Score')
    st.pyplot(plt)

def box_plots(data):
    st.subheader('Feature Distribution by Diagnosis')
    
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox(
        'Choose Feature', 
        features_to_plot,
        key='box_plot_feature_select'
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y=selected_feature, data=data, 
                palette=['lightgreen', 'lightcoral'])
    plt.title(f'{selected_feature} Distribution by Diagnosis')
    st.pyplot(plt)

def violin_plots(data):
    st.subheader('Violin Plots of Features')
    
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox(
        'Choose Feature', 
        features_to_plot,
        key='violin_plot_feature_select'
    )
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='diagnosis', y=selected_feature, data=data, 
                   palette=['lightgreen', 'lightcoral'], split=True)
    plt.title(f'{selected_feature} Distribution by Diagnosis')
    st.pyplot(plt)

if __name__ == '__main__':
    main()