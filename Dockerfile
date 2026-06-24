FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for fpdf2 and model
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "src/streamlit_dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
