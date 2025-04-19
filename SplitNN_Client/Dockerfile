# Use the official Python 3.12 image as the base image
FROM python:3.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install setuptools wheel && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the Python script to the working directory
COPY . /app/

# Set the default command to run the Python script
CMD ["python", "runner.py"]