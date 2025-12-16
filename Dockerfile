# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files
# PYTHONUNBUFFERED: Ensures non-buffered output for real-time logs in Cloud Run
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (none currently needed for this stack, but good practice)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Cloud Run sets this env var automatically, default 8080)
ENV PORT=8080

# Run the application
# We use 'exec' to ensure uvicorn receives signals correctly
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
