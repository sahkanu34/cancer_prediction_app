import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.feature_selection import SelectKBest, f_classif

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
        
    

        # Interactive pie chart using plotly
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=['Benign', 'Malignant'],
            values=prediction_proba[0],
            hole=.3,
            marker_colors=['lightgreen', 'lightcoral']
        )])
        
        fig.update_layout(
            title="Diagnosis Distribution",
            annotations=[dict(text=f'{max(prediction_proba[0]):.1%}', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig)

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
        st.metric("Accuracy", "97%")
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
    
    # Calculate diagnosis counts
    diagnosis_counts = data['diagnosis'].value_counts()
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=('Diagnosis Proportion', 'Diagnosis Counts'))
    
    # Add pie chart
    fig.add_trace(
        go.Pie(labels=['Benign', 'Malignant'],
               values=diagnosis_counts.values,
               textinfo='percent',
               hovertemplate="<b>%{label}</b><br>" +
                           "Count: %{value}<br>" +
                           "Percentage: %{percent}<extra></extra>",
               marker_colors=['lightgreen', 'lightcoral']),
        row=1, col=1
    )
    
    # Add bar chart
    fig.add_trace(
        go.Bar(x=['Benign', 'Malignant'],
               y=diagnosis_counts.values,
               text=diagnosis_counts.values,
               textposition='auto',
               marker_color=['lightgreen', 'lightcoral'],
               hovertemplate="<b>%{x}</b><br>" +
                           "Count: %{y}<extra></extra>"),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        height=500,
        width=1000,
        title_text="Tumor Diagnosis Distribution",
        yaxis_title="Number of Cases",
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

def feature_distributions(data):
    st.subheader('Feature Distributions')
    
    # Remove non-feature columns
    features_to_plot = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns.tolist()
    
    # Create selectbox for feature selection
    selected_feature = st.selectbox(
        'Choose Feature to Visualize', 
        features_to_plot,
        key='feature_distribution_select'
    )
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Histogram of {selected_feature}',
                                      f'Density Plot of {selected_feature}'))
    
    # Calculate histogram data
    hist_values, bin_edges = np.histogram(data[selected_feature], bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate KDE for density plot
    kde = gaussian_kde(data[selected_feature])
    x_range = np.linspace(data[selected_feature].min(), data[selected_feature].max(), 200)
    density_values = kde(x_range)
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data[selected_feature],
            nbinsx=30,
            name='Histogram',
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add density plot
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=density_values,
            name='Density',
            fill='tozeroy',
            hovertemplate="Value: %{x}<br>Density: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1000,
        showlegend=False,
        title_text=f"Distribution of {selected_feature}",
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

def correlation_heatmap(data):
    st.subheader('Feature Correlation Heatmap')
    
    # Calculate correlation matrix
    correlation_matrix = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Correlation Matrix of Diagnostic Features',
        width=1000,
        height=800,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    # Make the heatmap square
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    
    # Display the plot
    st.plotly_chart(fig)

def feature_importance(data):
    st.subheader('Feature Importance Analysis')
    
    # Prepare data
    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(data['diagnosis'])
    
    # Calculate feature importance
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=feature_scores.head(5)['score'],
            y=feature_scores.head(5)['feature'],
            orientation='h',
            marker=dict(
                color=feature_scores.head(5)['score'],
                colorscale='Viridis'
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'F-Score: %{x:.2f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Top 5 Most Important Features',
        xaxis_title='F-Score',
        yaxis_title='Feature',
        width=800,
        height=500,
        yaxis={'autorange': 'reversed'}  # To match the original order
    )
    
    # Display the plot
    st.plotly_chart(fig)

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