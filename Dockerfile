# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Flask/Gunicorn port
EXPOSE 8000

# Default command to run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--worker-class", "gevent", "--timeout", "180", "main:app"]
