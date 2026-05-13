FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
COPY needle-repo /app/needle-repo
# Skip heavy dependencies during build if possible or use a more lightweight approach
# Actually, the dependencies are necessary for Needle.
# I will just ensure the build context is as small as possible.
RUN pip install --no-cache-dir -e /app/needle-repo
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
CMD ["python", "webapp.py"]
