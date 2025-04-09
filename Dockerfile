# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container (optional for APIs, not needed here)
EXPOSE 5000

# Define environment variable (optional)

# Run the application when the container starts
CMD ["python", "bayesian.py"]
