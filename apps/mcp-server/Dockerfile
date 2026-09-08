FROM python:3.11-slim

WORKDIR /app

# Copia primeiro apenas os requisitos para aproveitar o cache do Docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100

CMD ["python", "server.py"]
