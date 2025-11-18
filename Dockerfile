# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run training inside container to ensure model exists (optional, or copy local .pth)
RUN python train.py

# Expose port
EXPOSE 5000

# Command to run app
CMD ["python", "app.py"]