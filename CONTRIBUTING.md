# Contribuindo com o PredictIQ

Guia prático para rodar, testar e modificar o projeto. Todos os comandos
abaixo foram executados e verificados neste repositório — nenhum é
hipotético. Para o contexto arquitetural completo, veja o [README](README.md);
para as diretrizes de estilo de código, veja o [CLAUDE.md](CLAUDE.md).

## 1. Setup inicial

```bash
git clone <repo>
cd projeto-tcc

# Dois arquivos de ambiente distintos — os dois são obrigatórios (ver README §7)
cp .env.example .env
cp apps/frontend/.env.local.example apps/frontend/.env.local

# Dataset MetroPT-3 (necessário antes da 1ª execução — ver README §12)
# baixe o CSV da UCI e coloque em apps/ml/data/raw/, depois:
cd apps/ml && python -m src.ingest_metropt && cd ../..
```

Edite `.env` com credenciais reais do Postgres (o exemplo usa placeholders).
**Nunca** copie valores reais de `.env` para `.env.example` ou
`apps/frontend/.env.local.example` — ambos são versionados no Git.

## 2. Rodando o projeto

### Via Docker (forma principal)

```bash
docker compose up --build
```

Serviços: `nginx` (porta 80, entrypoint único), `api`, `frontend`, `db`
(Postgres), `mlflow` (porta 5000), `ml` (Jupyter, rede interna). Detalhes em
[README §8](README.md#8-como-rodar-docker-compose).

```bash
docker compose ps                    # status dos containers
docker compose logs api --tail=50    # logs de um serviço
docker compose restart frontend      # necessário após editar componentes —
                                      # o Turbopack não recarrega de forma
                                      # confiável via bind mount no Windows
```

### Backend sem Docker

```bash
cd apps/backend
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Precisa de `DATABASE_URL` apontando para `localhost` (não `db`) — ver
[README §9.2](README.md#92-rodando-alembic-localmente-fora-do-docker).

### Frontend sem Docker

```bash
cd apps/frontend
corepack enable
pnpm install
pnpm dev
```

Acesse `http://localhost:3000`. Confirme que `apps/frontend/.env.local`
existe (passo 1) — sem ele a aplicação lança
`NEXT_PUBLIC_API_URL não está definida` em runtime.

## 3. Testes

```bash
# Backend (pytest + SQLite em memória — não exige Postgres)
cd apps/backend && pytest tests/ -v
# ou via Docker:
docker compose exec api pytest tests/ -v

# ML (pytest)
cd apps/ml && pytest tests/ -v

# Frontend (Vitest)
cd apps/frontend && pnpm test
# ou via Docker:
docker compose exec frontend pnpm exec vitest run

# Frontend E2E (Playwright)
cd apps/frontend
pnpm exec playwright install   # primeira vez
pnpm e2e
```

Ver [README §14](README.md#14-testes) para cobertura detalhada, load test
(RNF-37) e memory-leak check (RNF-38).

> **Nota**: no momento desta revisão, 23 testes Vitest falham por um
> problema pré-existente e conhecido (limiares de `RISK_THRESHOLDS`
> desalinhados com fixtures de teste — não relacionado a mudanças recentes).
> Não é esperado que você "corrija" esses testes ao trabalhar em outra
> feature; se sua mudança introduzir *novas* falhas, isso é sim seu.

## 4. Lint, typecheck e build

```bash
# Backend
cd apps/backend
ruff check .
mypy src/ --ignore-missing-imports
black --check .

# Frontend
cd apps/frontend
pnpm run lint          # eslint
pnpm exec tsc --noEmit # typecheck
pnpm run build         # build de produção (next build)
```

O repositório usa `pre-commit` (ver `.pre-commit-config.yaml`): `ruff`,
`black`, `mypy`, `eslint`, `prettier` rodam automaticamente em `git commit`.
Se um hook reformatar arquivos (black/prettier), `git add` de novo antes de
recommitar — o commit é abortado na primeira passada só para dar chance de
revisar o diff formatado.

## 5. Fluxo de Git

- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`, `test:`) — regra explícita
  em `CLAUDE.md` §6.
- **Branches**: nomeie por tipo e escopo (`feat/nome-da-feature`,
  `fix/bug-especifico`, `docs/assunto`, `perf/otimizacao`) — mantém o
  histórico legível quando várias frentes de trabalho coexistem.
- Não crie um fluxo de squash/rebase obrigatório nem regras de review que
  não existam hoje no projeto — este é um repositório de TCC com poucos
  contribuidores; mantenha o processo simples.

## 6. Cuidados específicos — SSE e WebSocket

Antes de tocar em `routers/stream.py`, `routers/alerts_ws.py`,
`sensor_stream_service.py`, `ws_manager.py`, `hooks/use-sse.ts` ou
`hooks/use-alert-websocket.ts`, leia as seções 4.3 (SSE) e 4.4 (WebSocket) do
[README, §4](README.md#4-fluxo-de-dados-em-tempo-real)
— eles documentam o comportamento real (formato de evento, heartbeat,
reconexão, timeouts) que qualquer mudança precisa preservar. Em particular:

- **Não altere o protocolo** (formato de frame SSE/WS) sem atualizar os
  DOIS lados (backend E os hooks do frontend) — eles não têm um schema
  compartilhado, a paridade é mantida manualmente.
- **Teste sempre através do Nginx**, não só direto na API — vários
  problemas (buffering, headers de upgrade) só aparecem atrás do proxy.
  Ver os comandos de diagnóstico em README §4.3/§4.4.
- Se mudar algo no `infra/nginx/nginx.conf`, valide manualmente que SSE e
  WS continuam funcionando via `curl -N` (SSE) e o handshake com
  `Upgrade: websocket` (WS) — não há teste automatizado cobrindo a
  configuração do Nginx em si.

## 7. Performance

Scripts de benchmark na raiz do repo (não hipotéticos — rodados e
documentados em `frontend_performance_report.md` e no README §14.5-14.8):

- `locust_streaming.py` — carga SSE concorrente (RNF-37).
- `check_memory_leak.py` — crescimento de RSS sob carga (RNF-38).
- `apps/frontend/scripts/measure-perf.mjs` — FPS/long-tasks reais via
  Playwright (RNF-39).
- `apps/backend/benchmark_models.py` — comparação F1/latência/memória
  entre os modelos de ML (RF-18).

Não adicione otimizações (memoização, virtualização, code-splitting) sem
medir o gargalo real primeiro — ver a justificativa de cada otimização já
aplicada em `frontend_performance_report.md` como referência de nível de
evidência esperado.
