FROM python:3.11-slim

WORKDIR /app

# RNF-62 — a base `python:3.11-slim` traz pip 24.0 / setuptools 79.x, com
# CVEs conhecidos (pip: PYSEC-2026-196/1795/1796/2875/2876/3721; setuptools:
# PYSEC-2026-3447). Não são dependências da aplicação (não são importados em
# runtime), mas ficam na imagem — atualizar aqui zera o achado do pip-audit
# sem tocar o código.
RUN pip install --no-cache-dir --upgrade "pip>=26.2" "setuptools>=83.0"

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