# Use the official Python 3.12 image as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script to the working directory
COPY runner.py .

# Set the default command to run the Python script
CMD ["python", "runner.py"]