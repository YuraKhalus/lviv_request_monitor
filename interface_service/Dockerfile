# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app/app

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# --server.port is necessary to ensure it runs on the exposed port
# --server.address 0.0.0.0 makes it accessible from outside the container
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
