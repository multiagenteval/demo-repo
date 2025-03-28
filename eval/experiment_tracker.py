from pathlib import Path
import json
import yaml
from datetime import datetime
import git
from typing import Dict

class ExperimentTracker:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def save_experiment(self, metrics: dict, config: dict):
        # Get git commit info
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        commit_msg = repo.head.object.message
        
        # Create experiment record
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "commit_hash": commit_hash,
            "commit_message": commit_msg,
            "metrics": metrics,
            "config": config
        }
        
        # Save with commit hash in filename
        result_file = self.results_dir / f"exp_{commit_hash[:8]}.json"
        with open(result_file, "w") as f:
            json.dump(experiment, f, indent=2)
        
        return result_file

    def load_baseline(self) -> dict:
        """Load the baseline metrics from the main branch"""
        try:
            repo = git.Repo(search_parent_directories=True)
            main_branch = repo.refs["main"]
            baseline_commit = main_branch.commit
            
            # Find experiment from baseline commit
            baseline_file = list(self.results_dir.glob(f"exp_{baseline_commit.hexsha[:8]}*.json"))
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
                baseline_val = baseline_metrics["metrics"].get(key)
                
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