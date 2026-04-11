FROM node:20

WORKDIR /app
COPY . .

RUN npm install -g pnpm
RUN pnpm install

CMD ["pnpm", "dev"]