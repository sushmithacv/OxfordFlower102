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

# Create .streamlit directory and copy the configuration files
RUN mkdir -p ~/.streamlit && \
    echo "[general]\nemail = \"youremail@example.com\"" > ~/.streamlit/credentials.toml && \
    echo "[server]\nport = 8501\nenableCORS = false\nheadless = true\n" > ~/.streamlit/config.toml

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Optionally, add a health check
HEALTHCHECK CMD curl --fail http://localhost:8501/ || exit 1

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
