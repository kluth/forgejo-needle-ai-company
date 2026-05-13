FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Install Needle
COPY needle-repo /app/needle-repo
RUN pip install -e /app/needle-repo

# Install Webapp dependencies
COPY orchestrator/requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

CMD ["python", "webapp.py"]
