from eval.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
commit = tracker.repo.head.commit
tracker.update_pending_metrics(commit.hexsha, commit.message) 