FROM python:3.9-slim

WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install pytest playwright requests \
    && playwright install --with-deps

# Expose Streamlit port
EXPOSE 8501

# Default command (for local dev)
CMD ["streamlit", "run", "index.py", "--server.headless", "true", "--server.port", "8501"]
