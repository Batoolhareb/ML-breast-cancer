FROM python:3.9-slim

WORKDIR /machine-learning

COPY requirements.txt .

# Upgrade pip and set trusted sources to avoid mirror hash mismatch
RUN pip install --upgrade pip \
 && pip config set global.timeout 100 \
 && pip install --no-cache-dir -r requirements.txt

# Copy everything else AFTER dependencies to leverage Docker cache
COPY . .

# Create required directories
RUN mkdir -p dataset output model

CMD ["python", "model.py"]
