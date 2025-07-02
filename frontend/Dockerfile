FROM python:3.9-slim

WORKDIR /frontend

COPY requirements.txt . 
RUN pip install --upgrade pip \
    && pip install -r requirements.txt 

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "index.py"]