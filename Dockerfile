# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Copy requirements and install dependencies
COPY dashboard_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY dashboard.py .

# Create a minimal working metrics file
RUN echo '[{"timestamp": "2025-03-28T18:52:03.972661", "git": {"commit_hash": "ad6013a5", "commit_message": "Test commit"}, "metrics_by_dataset": {"test": {"accuracy": 0.85, "f1": 0.84, "precision": 0.83, "recall": 0.82}}}]' > metrics_history.json

# Expose the port Cloud Run expects
ENV PORT=8080
EXPOSE ${PORT}

# Command to run the application
CMD streamlit run dashboard.py --server.port=${PORT} --server.address=0.0.0.0 