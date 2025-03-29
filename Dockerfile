FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only the required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY dashboard_requirements.txt .
RUN pip install --no-cache-dir -r dashboard_requirements.txt

# Copy only the necessary files
COPY dashboard.py .
COPY metrics_history.json .

# Set the metrics file path
ENV METRICS_FILE=/app/metrics_history.json

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the dashboard with optimized settings
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.maxUploadSize=0", "--browser.serverAddress=0.0.0.0"] 