FROM python:3.10

WORKDIR /backend

# Set pip configuration
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10

# Use a reliable PyPI mirror (Tsinghua University)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]