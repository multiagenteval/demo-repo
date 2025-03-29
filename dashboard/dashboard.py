import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

class MetricsStore:
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.getenv('METRICS_FILE', 'metrics_history.json')
        
    def get_available_models(self) -> List[str]:
        """Get list of available models from metrics history"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            return list(set(entry.get('model_name', 'default') for entry in data))
        except Exception as e:
            st.error(f"Error loading metrics data: {str(e)}")
            return []
            
    def get_metrics_for_model(self, model_name: str) -> pd.DataFrame:
        """Get metrics history for a specific model"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            processed_data = []
            for entry in data:
                if entry.get('model_name', 'default') != model_name:
                    continue
                    
                timestamp = datetime.fromisoformat(entry['timestamp'])
                commit_hash = entry['git']['commit_hash']
                commit_msg = entry['git']['commit_message'].split('\n')[0]
                
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

class ModelEvalDashboard:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.metrics_store = MetricsStore()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load model-specific configuration"""
        # This would typically load from a config file or database
        # For now, return a basic config
        return {
            "metrics": ["accuracy", "f1", "precision", "recall"],
            "datasets": ["test", "test_balanced", "test_adversarial"]
        }
        
    def calculate_aggregate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted aggregate score across metrics"""
        weights = {
            'accuracy': 0.4,
            'f1': 0.3,
            'precision': 0.15,
            'recall': 0.15
        }
        return sum(metrics[metric] * weight for metric, weight in weights.items())
        
    def render_overview(self, model_name: str):
        """Render the overview page"""
        st.header(f"Model Overview: {model_name}")
        
        # Load model metrics
        df = self.metrics_store.get_metrics_for_model(model_name)
        if df.empty:
            st.warning("No metrics data available for this model.")
            return
            
        # Latest metrics summary
        latest_metrics = df.groupby('dataset').last().reset_index()
        
        # Create metrics cards
        cols = st.columns(len(latest_metrics))
        for col, (_, row) in zip(cols, latest_metrics.iterrows()):
            with col:
                st.metric(
                    f"{row['dataset']} Accuracy",
                    f"{row['accuracy']:.3f}",
                    f"{row['accuracy'] - df[df['dataset'] == row['dataset']].iloc[-2]['accuracy']:+.3f}"
                )
                
        # Performance timeline
        st.subheader("Performance Timeline")
        fig = go.Figure()
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        colors = px.colors.qualitative.Set2
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[metric],
                name=metric.capitalize(),
                line=dict(color=colors[i], width=2),
                hovertemplate=f"{metric}: %{{y:.3f}}<br>Commit: %{{customdata}}<extra></extra>",
                customdata=df['commit_message']
            ))
            
        fig.update_layout(
            title="Metrics Timeline",
            xaxis_title="Timestamp",
            yaxis_title="Score",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def render_metrics(self, model_name: str):
        """Render detailed metrics page"""
        st.header(f"Detailed Metrics: {model_name}")
        
        df = self.metrics_store.get_metrics_for_model(model_name)
        if df.empty:
            st.warning("No metrics data available for this model.")
            return
            
        # Dataset selector
        selected_dataset = st.selectbox(
            "Select Dataset",
            df['dataset'].unique()
        )
        
        # Filter data for selected dataset
        dataset_df = df[df['dataset'] == selected_dataset]
        
        # Metrics table
        st.subheader("Latest Metrics")
        latest_metrics = dataset_df[['accuracy', 'f1', 'precision', 'recall']].iloc[-1]
        st.table(latest_metrics.round(3))
        
        # Performance trends
        st.subheader("Performance Trends")
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            fig = px.line(dataset_df, x='timestamp', y=metric)
            st.plotly_chart(fig, use_container_width=True)
            
    def render_commit_analysis(self, model_name: str):
        """Render commit analysis page"""
        st.header(f"Commit Analysis: {model_name}")
        
        df = self.metrics_store.get_metrics_for_model(model_name)
        if df.empty:
            st.warning("No metrics data available for this model.")
            return
            
        # Commit history with significant changes
        st.subheader("Significant Changes")
        
        for i in range(len(df)-1, 0, -1):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Calculate metric changes
            changes = {metric: current[metric] - previous[metric] 
                      for metric in ['accuracy', 'f1', 'precision', 'recall']}
            
            # Only show commits with significant changes
            if any(abs(change) > 0.01 for change in changes.values()):
                with st.expander(f"📝 {current['commit_hash'][:7]} - {current['commit_message']}"):
                    cols = st.columns(len(changes))
                    for col, (metric, change) in zip(cols, changes.items()):
                        col.metric(
                            metric.capitalize(),
                            f"{current[metric]:.3f}",
                            f"{change:+.3f}",
                            delta_color="normal"
                        )
                        
    def render_dashboard(self):
        """Main dashboard render function"""
        st.set_page_config(
            layout="wide",
            page_title="Model Evaluation Dashboard",
            page_icon="📊"
        )
        
        # Model selection
        available_models = self.metrics_store.get_available_models()
        if not available_models:
            st.error("No models found in metrics history.")
            return
            
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models
        )
        
        # Page selection
        page = st.sidebar.selectbox(
            "View",
            ["Overview", "Metrics", "Commit Analysis"]
        )
        
        # Render selected page
        if page == "Overview":
            self.render_overview(selected_model)
        elif page == "Metrics":
            self.render_metrics(selected_model)
        elif page == "Commit Analysis":
            self.render_commit_analysis(selected_model)

def main():
    # Load config from environment or default
    config_path = os.getenv('EVAL_CONFIG', 'eval-config.yaml')
    
    # Initialize and render dashboard
    dashboard = ModelEvalDashboard(config_path)
    dashboard.render_dashboard()

if __name__ == "__main__":
    main() 