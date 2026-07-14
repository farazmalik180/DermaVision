# Use official Python lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for opencv in utils)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend requirements (since Streamlit app uses the same backend logic/models)
COPY ../backend/requirements.txt ./backend_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r backend_requirements.txt

# Streamlit uses the root directory, so we copy both backend and streamlit_app
COPY ../backend ./backend
COPY ../streamlit_app ./streamlit_app

# Expose Streamlit port
EXPOSE 8501

# Run the streamlit application
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
