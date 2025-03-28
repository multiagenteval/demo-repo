import json
import pandas as pd
from pathlib import Path
import git
from typing import Dict, List, Optional
from datetime import datetime

class MetricsTracker:
    def __init__(self, metrics_dir: str = 'experiments/metrics'):
        self.metrics_dir = Path(metrics_dir)
        self.history_file = self.metrics_dir / 'metrics_history.json'
        self.csv_file = self.metrics_dir / 'metrics_history.csv'
        self.repo = git.Repo(search_parent_directories=True)
        
    def save_metrics(self, metrics: Dict, name: str):
        """Save metrics with git commit information"""
        commit = self.repo.head.commit
        
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'commit_hash': commit.hexsha,
            'commit_message': commit.message.strip(),
            'metrics': metrics,
            'run_name': name
        }
        
        # Load existing history
        history = self._load_history()
        history.append(metric_entry)
        
        # Save updated history
        self._save_history(history)
        
    def get_metric_history(self, metric_name: str) -> List[Dict]:
        """Get history of a specific metric across commits"""
        history = self._load_history()
        return [
            {
                'commit_hash': entry['commit_hash'][:8],
                'commit_message': entry['commit_message'].split('\n')[0],
                'value': entry['metrics'].get(metric_name),
                'timestamp': entry['timestamp']
            }
            for entry in history
            if metric_name in entry['metrics']
        ]
    
    def compare_with_previous(self, metrics: Dict) -> Dict:
        """Compare metrics with previous commit"""
        history = self._load_history()
        if not history:
            return {'status': 'first_commit'}
            
        previous = history[-1]['metrics']
        changes = {}
        
        for metric_name, current_value in metrics.items():
            if metric_name in previous:
                prev_value = previous[metric_name]
                if prev_value is not None and current_value is not None:
                    change = current_value - prev_value
                    changes[metric_name] = {
                        'current': current_value,
                        'previous': prev_value,
                        'change': change,
                        'change_percent': (change / prev_value) * 100 if prev_value != 0 else float('inf')
                    }
        
        return changes
    
    def _load_history(self) -> List:
        """Load metrics history from file"""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []
    
    def _save_history(self, history: List):
        """Save metrics history to file"""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def to_pandas(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame"""
        history = self._load_history()
        
        # Flatten the metrics data
        flat_data = []
        for entry in history:
            base_entry = {
                'timestamp': entry['timestamp'],
                'commit_hash': entry['commit_hash'][:8],
                'commit_message': entry['commit_message'].split('\n')[0],
                'run_name': entry['run_name']
            }
            # Add each metric as a separate column
            base_entry.update(entry['metrics'])
            flat_data.append(base_entry)
        
        df = pd.DataFrame(flat_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_csv(self):
        """Save metrics history as CSV"""
        df = self.to_pandas()
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_file, index=False)
        logger.info(f"Saved metrics history to {self.csv_file}")
    
    def plot_metric_history(self, metric_name: str, save_path: Optional[str] = None):
        """Plot history of a specific metric"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = self.to_pandas()
            
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # Plot metric over time
            plt.plot(df['timestamp'], df[metric_name], marker='o')
            
            plt.title(f'{metric_name} Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel(metric_name)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot metric history: {e}")

    def plot_all_metrics(self, save_dir: Optional[str] = None):
        """Plot history of all metrics"""
        df = self.to_pandas()
        metric_columns = [col for col in df.columns 
                        if col not in ['timestamp', 'commit_hash', 'commit_message', 'run_name']]
        
        for metric in metric_columns:
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f'{metric}_history.png'
            self.plot_metric_history(metric, save_path) 