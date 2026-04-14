FROM python:3.11-slim

WORKDIR /app

# Copia primeiro apenas os requisitos para aproveitar o cache do Docker
COPY requirements.txt .

# Instala TODAS as dependências do arquivo
RUN pip install --no-cache-dir -r requirements.txt

# Agora copia o resto do código
COPY . .

# Expõe a porta
EXPOSE 8000

# O comando agora vem do docker-compose, então não precisa de CMD aqui 
# ou pode deixar um padrão:
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]