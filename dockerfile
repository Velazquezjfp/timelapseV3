# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

#Create necessary working folders: 
RUN mkdir -p /app/photo

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask in the container
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Expose port 5000 for the Flask app
EXPOSE 8080

# Run the Flask app
#CMD ["python", "app.py"] not using this anymore

# Run the app with Gunicorn
CMD ["gunicorn", "--workers=4", "--timeout=120", "-b", "0.0.0.0:8080", "app:app"]