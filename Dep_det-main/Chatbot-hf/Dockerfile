# Use a lightweight Python CPU-based image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory and set permissions
RUN mkdir -p /app/cache && chmod -R 777 /app/cache
RUN apt-get update && apt-get install -y ffmpeg

# Copy code to container
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
