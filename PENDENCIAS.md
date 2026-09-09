# Pendências conhecidas

Itens implementados e validados com infraestrutura real, mas que dependem de
algo **fora do controle do código** (credenciais reais, decisão de produto,
ambiente de produção) para serem considerados 100% fechados. Não bloqueiam
merge/uso do sistema — documentados aqui para rastreio.

## RF-24 / RNF-48 — Notificações críticas via Telegram

- **Telegram Bot API real nunca foi testado.** Toda a suíte (`test_notifications.py`,
  35 testes) e toda a validação manual usam HTTP mockado — não existe
  `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` real neste ambiente de
  desenvolvimento. A lógica de negócio (threshold `>0.85`, rate limit,
  concorrência, orquestração, construção da Dashboard URL) foi validada
  contra infraestrutura real (Postgres real, classes reais) — só a chamada
  de rede ao Telegram em si permanece não verificada contra o serviço real.
  **Para fechar**: criar um bot via BotFather (passo a passo no
  [README §4.10](README.md)), preencher `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`
  no `.env`, provocar uma predição com `probability > 0.85` (real ou via
  script direto contra `get_alert_service()`) e confirmar o recebimento da
  mensagem no Telegram.

- **Rate limit via Postgres não expurga linhas expiradas em background.**
  Funciona corretamente no volume atual (1 linha por equipamento simulado)
  — a expiração é verificada na leitura (`WHERE expires_at <= now`), nunca
  limpa proativamente. Não escalaria da mesma forma para milhares de
  equipamentos sem um índice dedicado em `expires_at` e/ou uma rotina de
  limpeza periódica. Sem ação necessária no volume atual do projeto.

- **`DASHBOARD_URL` está com o default `http://localhost`.** Correto para
  desenvolvimento local; precisa ser trocado para o domínio real antes de
  qualquer deploy de produção, senão o link enviado no alerta do Telegram
  aponta para `localhost` do lado errado da rede.

## RNF-53 / RNF-54 — E2E/visual (Playwright + MSW)

- **`failure_alert.spec.ts`/`dashboard_flow.spec.ts` (pré-existentes,
  RNF-16/17) falham num container Playwright isolado e limpo** — as rotas
  relativas `/api/stream/sensors`/`/api/v1/predictions` não são
  interceptadas pelo MSW nesse método de validação (reproduzido sob Node 20
  e Node 24, sem alterar nenhuma linha desses specs). Não é uma regressão
  desta task — descoberto DURANTE a validação de `maintenance_assistant.spec.ts`
  (cujo teste RNF-53 depende do mesmo mecanismo no primeiro passo). **Para
  investigar**: confirmar se o runner real do GitHub Actions reproduz o
  mesmo problema (não verificado — só validado localmente/ad-hoc) e, se
  sim, determinar por que as rotas relativas do dashboard não casam com os
  handlers de `mocks/handlers.ts` nesse ambiente específico.
- Consequência direta do item acima: o teste
  `maintenance_assistant.spec.ts::RNF-53` falha de forma consistente no seu
  PRIMEIRO passo (banner crítico do Dashboard) até o achado acima ser
  corrigido — os outros 6 testes do mesmo arquivo passam 100% estáveis.

## RNF-50 / RNF-51 — Fila assíncrona de notificações (Celery + Redis)

- **Sem retry automático** (`max_retries=0`, deliberado) — uma falha de rede
  do worker não tenta de novo sozinha; o mecanismo de retry natural é a
  próxima predição crítica real do mesmo equipamento, depois que o rate
  limit é liberado. Reavaliar se o volume real de falhas transitórias
  justificar um retry com backoff.
- **Sem result backend** — o processo web nunca sabe se uma notificação
  específica foi entregue; só os logs do `celery-worker` mostram isso.
  Aceitável para o design fire-and-forget atual; adicionar um result
  backend (o mesmo Redis serviria) se essa visibilidade se tornar
  necessária no futuro.
- **`celery-worker` não espera a migração Alembic do `api` terminar** —
  `depends_on` no `docker-compose.yml` só espera Postgres/Redis saudáveis,
  não o comando interno de outro serviço. Irrelevante na prática (nenhuma
  task real é enfileirada antes do `api` terminar de subir), mas vale saber
  se o padrão de deploy mudar.
- **Telegram/Resend reais não testados através da fila** — mesma limitação
  já registrada acima para RF-24/RF-25 (sem credenciais reais neste
  ambiente); a validação real cobriu `producer -> Redis -> celery-worker ->
  task` e o não-bloqueio da inferência, não o envio efetivo.

## RF-25 / RNF-49 — Configuração de alertas (`/settings/alerts`)

- **Telegram e e-mail (Resend) reais nunca foram testados contra as APIs
  públicas.** Mesma limitação do RF-24 — nenhuma credencial real
  (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/`RESEND_API_KEY`) disponível
  neste ambiente. **Para fechar**: preencher as credenciais reais no
  `.env`, habilitar os dois canais em `/settings/alerts`, e clicar em
  "Testar Notificação" — confirmar recebimento em ambos.
- **`RESEND_FROM_EMAIL` usa um domínio placeholder** (`predictiq.dev`) —
  precisa ser trocado para um domínio verificado no Resend antes de
  qualquer envio real funcionar em produção.
- **Configuração é global (single-tenant), não por usuário** — decisão
  deliberada documentada no README §4.11 (o projeto não possui sistema de
  usuários). Se o projeto ganhar autenticação multiusuário no futuro,
  `alert_settings` precisará de uma migração para sair do modelo singleton.

## RF-20 / RNF-44 — Ingestão de manuais

- Imagem do `mcp-server` cresceu para ~10 GB porque `pip install
  sentence-transformers` puxa `torch` com dependências CUDA mesmo em
  container CPU-only. Otimização futura: instalar `torch` a partir do
  índice CPU-only da PyTorch (`--index-url
  https://download.pytorch.org/whl/cpu`).
- PDF sem texto extraível é reprocessado a cada execução da pipeline (custo
  baixo — só a extração via `pypdf`, nenhum embedding gerado).

## RF-22 / RNF-46 — Sugestão de manutenção (RAG)

- `apps/backend/requirements.txt` puxa `nvidia-nccl-cu12` (~340 MB) como
  dependência transitiva de `mcp`, sem uso real (backend é só cliente MCP).
  Infla a imagem sem necessidade — mesma classe de problema do torch/CUDA
  acima.

## RF-23 / RNF-47 — Streaming da sugestão (SSE)

- Llama 3.2 3B às vezes cita o placeholder interno `[MANUAL N]`
  literalmente no texto gerado, em vez de só usá-lo como referência mental.
  Não compromete veracidade/segurança do conteúdo — artefato de modelo
  pequeno seguindo formatação de forma imperfeita.
- Sem endpoint de arquivo estático para os PDFs dos manuais — por isso as
  referências no painel aparecem como texto, nunca como link clicável.

## Pré-existente (não introduzido por nenhuma das tasks acima)

- `apps/backend/tests/test_simulator.py` falha na coleta com `IndexError: 3`
  — confirmado via `git stash` como pré-existente antes da RF-22. Contorno
  usado em toda validação: `pytest --ignore=tests/test_simulator.py`.
