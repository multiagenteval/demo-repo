from pathlib import Path
import json
import yaml
from datetime import datetime
import git
from typing import Dict
import pandas as pd

class ExperimentTracker:
    def __init__(self):
        self.experiments_dir = Path('experiments/metrics')
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self.experiments_dir / 'metrics_history.json'
        self.csv_file = self.experiments_dir / 'metrics_history.csv'
        self.repo = git.Repo(search_parent_directories=True)

    def save_experiment(self, metrics_by_dataset: Dict[str, Dict], config: Dict, status: str = "pending") -> Dict:
        """Save experiment results with proper numpy value handling"""
        # Convert numpy values to Python native types
        processed_metrics = {}
        for dataset_name, metrics in metrics_by_dataset.items():
            processed_metrics[dataset_name] = {
                k: float(v) if hasattr(v, 'item') else v 
                for k, v in metrics.items()
            }

        experiment = {
            "timestamp": datetime.now().isoformat(),
            "git": {
                "commit_hash": "pending" if status == "pending" else self.repo.head.commit.hexsha,
                "commit_message": "Uncommitted changes" if status == "pending" else self.repo.head.commit.message.strip(),
                "branch": self.repo.active_branch.name,
                "status": status
            },
            "metrics_by_dataset": processed_metrics,
            "config": config
        }

        # Save with pretty printing for JSON
        with open(self.json_file, 'w') as f:
            json.dump(experiment, f, indent=2)

        # Save to CSV with proper value conversion
        self._save_csv(experiment)
        
        return experiment

    def load_baseline(self) -> dict:
        """Load the baseline metrics from the main branch"""
        try:
            repo = git.Repo(search_parent_directories=True)
            main_branch = repo.refs["main"]
            baseline_commit = main_branch.commit
            
            # Find experiment from baseline commit
            baseline_file = list(self.experiments_dir.glob(f"exp_{baseline_commit.hexsha[:8]}*.json"))
            if baseline_file:
                with open(baseline_file[0]) as f:
                    return json.load(f)
        except Exception as e:
            print(f"No baseline found: {e}")
        return None

    def compare_with_baseline(self, current_metrics: Dict) -> Dict:
        """Compare current metrics with baseline, handling None values"""
        try:
            baseline_metrics = self.load_baseline()
            if not baseline_metrics:
                return {'status': 'no_baseline'}
            
            comparisons = {}
            for key in current_metrics:
                current_val = current_metrics.get(key)
                baseline_val = baseline_metrics["metrics_by_dataset"].get(key)
                
                # Only compare if both values are not None
                if current_val is not None and baseline_val is not None:
                    diff = current_val - baseline_val
                    comparisons[key] = {
                        'current': current_val,
                        'baseline': baseline_val,
                        'diff': diff,
                        'diff_percent': (diff / baseline_val) * 100 if baseline_val != 0 else float('inf')
                    }
                else:
                    comparisons[key] = {
                        'current': current_val,
                        'baseline': baseline_val,
                        'diff': None,
                        'diff_percent': None,
                        'status': 'incomplete_data'
                    }
            
            return comparisons
            
        except Exception as e:
            print(f"Error comparing with baseline: {e}")
            return {'status': 'comparison_error', 'error': str(e)}

    def _save_json(self, experiment):
        # Save with commit hash in filename
        result_file = self.experiments_dir / f"exp_{experiment['git']['commit_hash'][:8]}.json"
        with open(result_file, "w") as f:
            json.dump(experiment, f, indent=2)

    def _save_csv(self, experiment):
        """Save experiment data to CSV including metrics for each dataset"""
        # Prepare the row data with fixed columns
        row_data = {
            'timestamp': experiment['timestamp'],
            'commit_hash': experiment['git']['commit_hash'],
            'commit_message': experiment['git']['commit_message'],
            'branch': experiment['git']['branch'],
            'status': experiment['git']['status']
        }

        # Add metrics for each dataset with dataset prefix
        for dataset_name, metrics in experiment['metrics_by_dataset'].items():
            for metric_name, value in metrics.items():
                col_name = f"{dataset_name}_{metric_name}"
                row_data[col_name] = float(value) if hasattr(value, 'item') else value

        # Create DataFrame with single row
        df_new = pd.DataFrame([row_data])
        
        try:
            if self.csv_file.exists():
                # Read existing CSV with explicit column names
                df_existing = pd.read_csv(self.csv_file)
                
                # Get union of all columns
                all_columns = sorted(list(set(df_existing.columns) | set(df_new.columns)))
                
                # Ensure both DataFrames have all columns
                for col in all_columns:
                    if col not in df_existing:
                        df_existing[col] = None
                    if col not in df_new:
                        df_new[col] = None
                
                # Ensure both DataFrames have same column order
                df_existing = df_existing[all_columns]
                df_new = df_new[all_columns]
                
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = df_new
            
            # Save with fixed columns
            df_final.to_csv(self.csv_file, index=False, na_rep='NA')
            
        except Exception as e:
            print(f"Error with CSV file, creating new one: {e}")
            # If there's any error, start fresh
            df_new.to_csv(self.csv_file, index=False, na_rep='NA')

    def _load_json(self, commit_hash):
        # Load from file
        result_file = self.experiments_dir / f"exp_{commit_hash[:8]}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return None 