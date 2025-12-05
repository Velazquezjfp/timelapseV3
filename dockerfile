# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, and unzip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create necessary working folders
RUN mkdir -p /app/photo

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (including gdown)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Download and extract yolov7_model from Google Drive
# File ID: 1o1NzJzR0ps8w0J0LAidr5eQf1DfBvT9j
RUN gdown --id 1o1NzJzR0ps8w0J0LAidr5eQf1DfBvT9j -O /app/yolov7_model.zip && \
    unzip /app/yolov7_model.zip -d /app && \
    rm /app/yolov7_model.zip

# Copy the rest of the application code
COPY . /app

# Expose port 8080 for the API
EXPOSE 8080

# Run the app with Gunicorn
CMD ["gunicorn", "--workers=4", "--timeout=120", "-b", "0.0.0.0:8080", "app:app"]
