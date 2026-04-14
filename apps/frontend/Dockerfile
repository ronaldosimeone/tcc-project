# Imagem Alpine: a mais leve e rápida para Node.js
FROM node:20-alpine

# Instala o pnpm de forma eficiente
RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro
# Isso evita que o 'pnpm install' rode de novo se você só mudar um componente UI
COPY package.json pnpm-lock.yaml* ./

# Instala as dependências (usa o cache do Docker)
RUN pnpm install

# Agora copia o resto do código do Next.js
COPY . .

# Porta padrão do Next.js
EXPOSE 3000

# Comando para rodar em modo desenvolvimento com Turbopack (se estiver configurado)
CMD ["pnpm", "dev"]