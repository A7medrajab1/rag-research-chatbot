FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only essential system deps
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install PyTorch CPU-only FIRST (saves ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (respects .dockerignore)
COPY . .

# Build index
RUN python -m src.pipeline --force

# Expose ports
EXPOSE 8000 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]