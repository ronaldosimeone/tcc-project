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
