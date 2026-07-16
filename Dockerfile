# Stage 1: Build the React Frontend
FROM node:20 AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Build the React app (outputs to /frontend/dist)
RUN npm run build

# Stage 2: Setup the Python Backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (needed for opencv)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend Python code
COPY backend/ ./backend/

# Copy the compiled React frontend into backend/static
COPY --from=frontend-builder /frontend/dist ./backend/static

# Expose port 7860 (Hugging Face Spaces requirement)
EXPOSE 7860

# Run Uvicorn server on port 7860
CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port 7860"]
