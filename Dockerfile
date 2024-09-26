# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to /app
COPY . .

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Create .streamlit directory and copy the configuration files
RUN mkdir -p ~/.streamlit

# Copy Streamlit configuration files
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
