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
COPY experiments/metrics/metrics_history.json .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"] 