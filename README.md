# AI Evaluation Ecosystem

A multi-agent evaluation ecosystem that plugs into your CI/CD process, evaluates your commits/releases, and provides bottom-up insight into your model performance across the board and longitudinally.

## Features

- **Commit Risk Analyzer**: Flagship agent that learns from historical commit-metric pairs to predict potential risks
- **Metric Evaluator**: Comprehensive model evaluation across multiple datasets and metrics
- **Test Suggester**: Recommends additional tests based on identified risks
- **Performance Tracker**: Longitudinal tracking of model performance
- **Interactive Dashboard**: Visualize model performance and commit impacts
- **CI/CD Integration**: Seamless integration with GitHub Actions, Jenkins, and GitLab CI

## Installation

```bash
pip install ai-eval-ecosystem
```

## Quick Start

1. Add a configuration file to your ML project:

```yaml
# .eval-config.yaml
model:
  name: "Your Model"
  type: "classification"
  metrics:
    - accuracy
    - f1
    - latency
  datasets:
    - name: "test"
      path: "data/test"

evaluation:
  schedule:
    - on_commit
    - nightly
  thresholds:
    accuracy: 0.95
    latency_ms: 100
```

2. Start the dashboard:

```bash
eval-dashboard --config .eval-config.yaml
```

3. Analyze commits:

```bash
eval-analyze --config .eval-config.yaml
```

4. Get test suggestions:

```bash
eval-suggest --config .eval-config.yaml
```

## Architecture

The ecosystem consists of several specialized agents:

1. **Commit Risk Analyzer**
   - Learns from historical commit-metric pairs
   - Predicts potential risks from new commits
   - Suggests additional tests

2. **Metric Evaluator**
   - Runs comprehensive model evaluations
   - Tracks multiple metrics across datasets
   - Generates detailed reports

3. **Test Suggester**
   - Recommends tests based on risks
   - Learns from historical test effectiveness
   - Prioritizes critical test cases

4. **Performance Tracker**
   - Tracks metrics over time
   - Identifies trends and anomalies
   - Generates performance reports

## Integration

### GitHub Actions

```yaml
# .github/workflows/eval.yml
name: Model Evaluation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install ai-eval-ecosystem
      - name: Run evaluation
        run: |
          eval-analyze --config .eval-config.yaml
          eval-suggest --config .eval-config.yaml
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Evaluate') {
            steps {
                sh 'pip install ai-eval-ecosystem'
                sh 'eval-analyze --config .eval-config.yaml'
                sh 'eval-suggest --config .eval-config.yaml'
            }
        }
    }
}
```

## Dashboard

The interactive dashboard provides:

- Model performance overview
- Detailed metrics analysis
- Commit impact analysis
- Test coverage insights
- Risk predictions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
