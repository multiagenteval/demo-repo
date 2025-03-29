# ML Model Demo Repository

This repository demonstrates how to integrate your ML project with the AI Evaluation Ecosystem. It shows a simple MNIST classifier with proper evaluation setup and CI/CD integration.

## Project Structure

```
.
├── models/              # Model definitions
├── train.py            # Training script
├── inference.py        # Inference script
├── data/               # Dataset directory
├── configs/            # Configuration files
├── tests/              # Test files
├── .eval-config.yaml   # Evaluation ecosystem config
└── .github/            # CI/CD workflows
```

## Integration with AI Evaluation Ecosystem

1. Install the evaluation ecosystem:
```bash
pip install ai-eval-ecosystem
```

2. Configure your project by creating `.eval-config.yaml`:
```yaml
model:
  name: "MNIST Classifier"
  type: "image_classification"
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

3. Add GitHub Actions workflow (`.github/workflows/eval.yml`):
```yaml
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
          pip install -r requirements.txt
      - name: Run evaluation
        run: |
          eval-analyze --config .eval-config.yaml
          eval-suggest --config .eval-config.yaml
```

## Local Development

1. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Run inference:
```bash
python inference.py
```

4. Run tests:
```bash
pytest
```

## Model Details

This demo implements a CNN classifier for MNIST digits with:
- Residual connections
- Batch normalization
- Dropout regularization
- Adversarial training support

## Metrics Tracking

The model tracks:
- Accuracy
- F1 Score
- Precision
- Recall
- Latency
- Memory usage

## CI/CD Integration

The repository includes:
- Automated testing
- Model evaluation on commits
- Performance monitoring
- Risk analysis
- Test suggestions

## License

MIT License
