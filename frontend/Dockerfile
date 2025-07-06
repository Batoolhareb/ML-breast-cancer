FROM python:3.9-slim

WORKDIR /frontend

# Copy application files
COPY . .

# Upgrade pip & install Python dependencies with increased timeout
RUN pip install --upgrade pip setuptools wheel \
    && pip install --timeout 100 --no-cache-dir -r requirements.txt

# Install Playwright browsers and system dependencies
RUN playwright install --with-deps

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "index.py", "--server.headless", "true", "--server.port", "8501"]
