FROM python:3.9-slim

WORKDIR /app

# Copy only the required files
COPY dashboard_requirements.txt requirements.txt
COPY dashboard.py .

# Create data directory and minimal metrics file
RUN mkdir -p data && \
    echo '[]' > data/metrics_history.json

# Install only the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"] 