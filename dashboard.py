import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np
import os
import sys

# Constants
METRICS_FILE = os.getenv('METRICS_FILE', 'metrics_history.json')
GITHUB_REPO = "multiagenteval/maee-demo-repo"  # Update this with your actual GitHub repo

# Model overview
MODEL_OVERVIEW = {
    "architecture": {
        "name": "Baseline CNN with Residual Connections",
        "description": """
        A convolutional neural network designed for MNIST digit classification with the following key features:
        
        - **Input Layer**: 1x28x28 (grayscale MNIST images)
        - **Convolutional Blocks**: 
            - Two main blocks with residual connections
            - Each block contains: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU
            - Skip connections preserve gradient flow
        - **Hidden Dimensions**: [32, 64] channels
        - **Dropout Rate**: 0.1 (for regularization)
        - **Output Layer**: 10 units (one for each digit)
        
        The model uses residual connections to improve gradient flow and feature preservation through the network.
        """,
        "training": """
        - **Optimizer**: Adam
        - **Learning Rate**: 0.001
        - **Batch Size**: 32
        - **Training Data**: MNIST training set
        - **Adversarial Training**: FGSM with ε=0.3
        """
    },
    "metrics": {
        "accuracy": """
        The proportion of correctly classified digits out of all predictions.
        - Range: 0.0 to 1.0
        - Higher is better
        - Most intuitive metric for overall performance
        """,
        "f1": """
        Harmonic mean of precision and recall.
        - Range: 0.0 to 1.0
        - Higher is better
        - Better than accuracy when classes are imbalanced
        - Penalizes both false positives and false negatives
        """,
        "precision": """
        The proportion of true positives out of all positive predictions.
        - Range: 0.0 to 1.0
        - Higher is better
        - Measures how many of the predicted positives are actually positive
        - Important when false positives are costly
        """,
        "recall": """
        The proportion of true positives out of all actual positives.
        - Range: 0.0 to 1.0
        - Higher is better
        - Measures how many of the actual positives are correctly identified
        - Important when false negatives are costly
        """
    }
}

# Dataset descriptions
DATASET_DESCRIPTIONS = {
    "test": {
        "name": "Standard Test Set",
        "description": "The standard MNIST test set containing 10,000 images. This is the primary evaluation dataset that provides a baseline measure of model performance on clean, unmodified data.",
        "interpretation": "High performance on this dataset indicates good basic recognition capabilities. However, high performance here alone doesn't guarantee robustness.",
        "use_case": "Baseline performance evaluation"
    },
    "test_balanced": {
        "name": "Balanced Test Set",
        "description": "A class-balanced version of the MNIST test set, ensuring equal representation of all digits. This helps identify any class imbalance issues in the model's predictions.",
        "interpretation": "Performance on this dataset reveals if the model has biases towards certain classes. A significant difference from the standard test set indicates class imbalance issues.",
        "use_case": "Class balance analysis"
    },
    "test_adversarial": {
        "name": "Adversarial Test Set",
        "description": "MNIST test set with FGSM (Fast Gradient Sign Method) adversarial attacks applied. This tests the model's robustness against carefully crafted perturbations.",
        "interpretation": "Performance here indicates the model's vulnerability to adversarial attacks. Lower performance suggests the model is more susceptible to adversarial examples.",
        "use_case": "Robustness evaluation"
    }
}

def load_data(file_path):
    """Load and process the metrics data with proper error handling"""
    try:
        # Try absolute path first
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # Try relative to current directory
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        
        if not os.path.exists(full_path):
            st.error(f"Metrics file not found at: {full_path}")
            st.info("Please ensure the metrics file is in the correct location or set the METRICS_FILE environment variable.")
            return pd.DataFrame()
            
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Process the data into a more usable format
        processed_data = []
        for entry in data:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            commit_hash = entry['git']['commit_hash']
            commit_msg = entry['git']['commit_message'].split('\n')[0]  # First line
            
            for dataset, metrics in entry['metrics_by_dataset'].items():
                metrics_row = {
                    'timestamp': timestamp,
                    'commit_hash': commit_hash,
                    'commit_message': commit_msg,
                    'dataset': dataset,
                    **metrics
                }
                processed_data.append(metrics_row)
        
        return pd.DataFrame(processed_data)
    except Exception as e:
        st.error(f"Error loading metrics data: {str(e)}")
        return pd.DataFrame()

def calculate_aggregate_score(metrics):
    """Calculate a weighted aggregate score across all metrics"""
    weights = {
        'accuracy': 0.4,
        'f1': 0.3,
        'precision': 0.15,
        'recall': 0.15
    }
    return sum(metrics[metric] * weight for metric, weight in weights.items())

def get_github_commit_url(commit_hash):
    """Generate GitHub commit URL"""
    return f"https://github.com/{GITHUB_REPO}/commit/{commit_hash}"

def main():
    # Main dashboard
    st.set_page_config(
        layout="wide",
        page_title="ML Experiment Metrics Dashboard",
        page_icon="📊"
    )
    
    st.title("ML Experiment Metrics Dashboard")
    
    # Model Overview Section
    st.header("Model Overview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Architecture")
        st.markdown(MODEL_OVERVIEW["architecture"]["description"])
        st.subheader("Training Configuration")
        st.markdown(MODEL_OVERVIEW["architecture"]["training"])
    
    with col2:
        st.subheader("Performance Metrics")
        for metric, description in MODEL_OVERVIEW["metrics"].items():
            with st.expander(f"{metric.capitalize()}"):
                st.markdown(description)
    
    # Load data
    df = load_data(METRICS_FILE)
    
    # Check if we have any data
    if df.empty:
        st.warning("No metrics data available. Please ensure the metrics file is properly populated.")
        return
        
    # Add aggregate score
    df['aggregate_score'] = df.apply(
        lambda x: calculate_aggregate_score(x[['accuracy', 'f1', 'precision', 'recall']]),
        axis=1
    )
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Select Dataset",
        df['dataset'].unique(),
        format_func=lambda x: DATASET_DESCRIPTIONS[x]['name']
    )
    
    # Display dataset description
    st.markdown(f"""
    ### {DATASET_DESCRIPTIONS[selected_dataset]['name']}
    
    **Description:** {DATASET_DESCRIPTIONS[selected_dataset]['description']}
    
    **Use Case:** {DATASET_DESCRIPTIONS[selected_dataset]['use_case']}
    
    **Interpretation Guide:** {DATASET_DESCRIPTIONS[selected_dataset]['interpretation']}
    """)
    
    # Filter data for selected dataset
    dataset_df = df[df['dataset'] == selected_dataset]
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main metrics timeline
        fig = go.Figure()
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        colors = px.colors.qualitative.Set2
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=dataset_df['timestamp'],
                y=dataset_df[metric],
                name=metric.capitalize(),
                line=dict(color=colors[i], width=2),
                hovertemplate=f"{metric}: %{{y:.3f}}<br>Commit: %{{customdata}}<extra></extra>",
                customdata=dataset_df['commit_message']
            ))
        
        fig.update_layout(
            title=f"Metrics Timeline - {DATASET_DESCRIPTIONS[selected_dataset]['name']}",
            xaxis_title="Timestamp",
            yaxis_title="Score",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Aggregate score card
        latest_score = dataset_df['aggregate_score'].iloc[-1]
        score_change = latest_score - dataset_df['aggregate_score'].iloc[-2]
        
        st.metric(
            "Aggregate Performance Score",
            f"{latest_score:.3f}",
            f"{score_change:+.3f}",
            delta_color="normal"
        )
        
        # Latest metrics table
        latest_metrics = dataset_df[['accuracy', 'f1', 'precision', 'recall']].iloc[-1]
        st.table(latest_metrics.round(3))
    
    # Commit history and changes
    st.subheader("Commit History and Significant Changes")
    
    for i in range(len(dataset_df)-1, 0, -1):
        current = dataset_df.iloc[i]
        previous = dataset_df.iloc[i-1]
        
        # Calculate metric changes
        changes = {metric: current[metric] - previous[metric] 
                  for metric in ['accuracy', 'f1', 'precision', 'recall']}
        
        # Only show commits with significant changes
        if any(abs(change) > 0.01 for change in changes.values()):
            commit_url = get_github_commit_url(current['commit_hash'])
            with st.expander(f"📝 [{current['commit_hash'][:7]}]({commit_url}) - {current['commit_message']}"):
                cols = st.columns(len(changes))
                for col, (metric, change) in zip(cols, changes.items()):
                    col.metric(
                        metric.capitalize(),
                        f"{current[metric]:.3f}",
                        f"{change:+.3f}",
                        delta_color="normal"
                    )
    
    # Cross-dataset comparison
    st.subheader("Cross-Dataset Performance Comparison")
    latest_data = df.groupby('dataset').last().reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            name=metric.capitalize(),
            x=[DATASET_DESCRIPTIONS[d]['name'] for d in latest_data['dataset']],
            y=latest_data[metric],
            text=latest_data[metric].round(3),
            textposition='auto',
        ) for metric in metrics
    ])
    
    fig.update_layout(
        barmode='group',
        title="Latest Metrics Across All Datasets",
        xaxis_title="Dataset",
        yaxis_title="Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 