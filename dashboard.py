import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np

# Load and process the data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process the data into a more usable format
    processed_data = []
    for entry in data:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        commit_hash = entry['git']['commit_hash'][:7]  # Short hash
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

def calculate_aggregate_score(metrics):
    """Calculate a weighted aggregate score across all metrics"""
    weights = {
        'accuracy': 0.4,
        'f1': 0.3,
        'precision': 0.15,
        'recall': 0.15
    }
    return sum(metrics[metric] * weight for metric, weight in weights.items())

# Main dashboard
st.set_page_config(layout="wide")
st.title("ML Experiment Metrics Dashboard")

# Load data
df = load_data('experiments/metrics/metrics_history.json')

# Add aggregate score
df['aggregate_score'] = df.apply(
    lambda x: calculate_aggregate_score(x[['accuracy', 'f1', 'precision', 'recall']]),
    axis=1
)

# Dataset selector
selected_dataset = st.selectbox(
    "Select Dataset",
    df['dataset'].unique(),
    format_func=lambda x: x.replace('_', ' ').title()
)

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
        title=f"Metrics Timeline - {selected_dataset.replace('_', ' ').title()}",
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
        with st.expander(f"📝 {current['commit_message']} ({current['commit_hash']})"):
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
        x=latest_data['dataset'],
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