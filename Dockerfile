FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync --no-dev

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/extracted data/ground_truth

# Expose Streamlit port
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
