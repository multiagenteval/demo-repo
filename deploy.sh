#!/bin/bash

# Exit on error
set -e

# Get project ID from gcloud config
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No project ID set. Run 'gcloud config set project YOUR_PROJECT_ID' first."
    exit 1
fi

# Build and push the container
echo "Building and pushing container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/metrics-dashboard --timeout=600

# Deploy to Cloud Run with minimal configuration
echo "Deploying to Cloud Run..."
gcloud run deploy metrics-dashboard \
    --image gcr.io/$PROJECT_ID/metrics-dashboard \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --no-cpu-throttling \
    --execution-environment gen2

# Get the service URL
SERVICE_URL=$(gcloud run services describe metrics-dashboard --platform managed --region us-central1 --format='value(status.url)')
echo "Deployment complete! Your dashboard is available at: $SERVICE_URL" 