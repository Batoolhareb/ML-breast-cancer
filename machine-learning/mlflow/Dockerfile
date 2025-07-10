FROM python:3.9-slim

WORKDIR /mlflow

COPY . .

RUN pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt

RUN mkdir -p dataset output model

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns"]
