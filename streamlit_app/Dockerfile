# Use official Python lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies are not needed because we use opencv-python-headless

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
