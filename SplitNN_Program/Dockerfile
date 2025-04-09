
# Use the official Python 3.13 slim base image
FROM python:3.13

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/
RUN apt install
# Install dependencies
RUN pip install setuptools wheel && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
RUN pip install gunicorn
# Copy the remaining application code into the container
COPY . /app/


# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "SplitNN_Program.wsgi"]