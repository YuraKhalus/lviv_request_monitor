# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
# Using --no-cache-dir for a smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
