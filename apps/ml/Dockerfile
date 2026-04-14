# Usando a imagem slim para economizar uns 600MB
FROM python:3.11-slim

# Instala dependências de sistema para compilar pacotes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o seu requirements gigante
COPY requirements.txt .

# Instala tudo de uma vez
RUN pip install --no-cache-dir -r requirements.txt

# Copia seus notebooks e a pasta de dados
COPY . .

# Porta padrão do Jupyter
EXPOSE 8888

# Comando para rodar o Jupyter aceitando conexões de fora do container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]