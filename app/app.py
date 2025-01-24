import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset and model
DATA_PATH = "data.csv"
MODEL_PATH = "breast_cancer_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

def main():
    st.image('img.jpeg', use_column_width=True)
    st.title('Breast Cancer Diagnosis Predictor')
    
    # Sidebar navigation
    st.sidebar.header('Navigation')
    app_mode = st.sidebar.selectbox('Choose Module', 
        ['Prediction', 'Model Details', 'Data Insights'])
    
    # Load model once
    model = load_model()
    data = load_dataset()
    
    # Routing
    if app_mode == 'Prediction':
        prediction_page(model, data)
    elif app_mode == 'Model Details':
        model_info_page(model)
    else:
        dataset_exploration_page(data)

def prediction_page(model, data):
    st.header('Tumor Diagnosis Prediction')
    
    # Get all feature names
    feature_names = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    
    # Dynamic feature input
    st.sidebar.header('Diagnostic Parameters')
    input_features = {}
    
    for feature in feature_names:
        # Dynamically calculate min, max based on data
        min_val = data[feature].min()
        max_val = data[feature].max()
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
    
    # Performance metrics column
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "96%")
    with col2:
        st.metric("Precision", "94%")

def dataset_exploration_page(data):
    st.header('Advanced Data Insights')
    
    # Visualization selection
    viz_option = st.selectbox('Choose Visualization', [
        'Overview', 
        'Diagnosis Distribution', 
        'Feature Distributions', 
        'Correlation Heatmap',
        'Feature Importance',
        'Box Plots by Diagnosis',
        'Violin Plots'
    ])
    
    if viz_option == 'Overview':
        dataset_overview(data)
    elif viz_option == 'Diagnosis Distribution':
        diagnosis_distribution(data)
    elif viz_option == 'Feature Distributions':
        feature_distributions(data)
    elif viz_option == 'Correlation Heatmap':
        correlation_heatmap(data)
    elif viz_option == 'Feature Importance':
        feature_importance(data)
    elif viz_option == 'Box Plots by Diagnosis':
        box_plots(data)
    else:
        violin_plots(data)

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
    
    # Pie Chart
    diagnosis_counts = data['diagnosis'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie Chart
    ax1.pie(diagnosis_counts, labels=['Benign', 'Malignant'], autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Diagnosis Proportion')
    
    # Bar Chart
    diagnosis_counts.plot(kind='bar', ax=ax2, color=['lightgreen', 'lightcoral'])
    ax2.set_title('Diagnosis Counts')
    ax2.set_xlabel('Diagnosis')
    ax2.set_ylabel('Number of Cases')
    
    st.pyplot(fig)

def feature_distributions(data):
    st.subheader('Feature Distributions')
    
    # Select features for visualization
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox('Choose Feature to Visualize', features_to_plot)
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    data[selected_feature].hist(ax=ax1, bins=30)
    ax1.set_title(f'Histogram of {selected_feature}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    # Density Plot
    data[selected_feature].plot(kind='density', ax=ax2)
    ax2.set_title(f'Density Plot of {selected_feature}')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    
    st.pyplot(fig)

def correlation_heatmap(data):
    st.subheader('Feature Correlation Heatmap')
    
    # Prepare correlation matrix
    correlation_matrix = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Diagnostic Features')
    st.pyplot(plt)

def feature_importance(data):
    st.subheader('Feature Importance Analysis')
    
    # Use SelectKBest to rank features
    from sklearn.feature_selection import f_classif, SelectKBest
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(data['diagnosis'])
    
    # Select top features
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    
    # Create feature importance plot
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
    
    # Select feature to plot
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox('Choose Feature', features_to_plot)
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y=selected_feature, data=data, 
                palette=['lightgreen', 'lightcoral'])
    plt.title(f'{selected_feature} Distribution by Diagnosis')
    st.pyplot(plt)

def violin_plots(data):
    st.subheader('Violin Plots of Features')
    
    # Select feature to plot
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    selected_feature = st.selectbox('Choose Feature', features_to_plot)
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='diagnosis', y=selected_feature, data=data, 
                   palette=['lightgreen', 'lightcoral'], split=True)
    plt.title(f'{selected_feature} Distribution by Diagnosis')
    st.pyplot(plt)

if __name__ == '__main__':
    main()