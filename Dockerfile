FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and metrics
COPY dashboard.py .
COPY experiments/ experiments/

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"] 