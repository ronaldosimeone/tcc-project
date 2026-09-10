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

## RNF-53 / RNF-54 — E2E/visual (Playwright + MSW) — RESOLVIDO

~~`failure_alert.spec.ts`/`dashboard_flow.spec.ts` falham num container
Playwright isolado~~ — **corrigido** (task "fix de integração MSW/Playwright
do Dashboard"). Causa raiz real, composta por 3 defeitos independentes —
não só o mismatch de URL suspeitado originalmente:

1. `mocks/handlers.ts` registrava `/api/stream/sensors`/`/api/v1/predictions`
   na origem absoluta `API_BASE`, mas `hooks/use-sensor-data.ts` chama
   ambas com caminho relativo (resolvido contra a origem da própria
   página) — handler nunca casava, request ia pra rede real e recebia 404
   silenciosamente.
2. O handler de `/api/v1/predictions` sempre devolvia uma página vazia,
   independente do `__E2E_SCENARIO__` — `latest`/`riskLevel` nunca
   atualizava via polling.
3. **Achado maior**: `components/alert-panel.tsx` (testids `alert-panel`/
   `critical-banner` que esses specs usam) tinha sido desconectado de
   `SensorMonitor` no commit `de44dc1` ("immersive critical state"), muito
   antes desta task — código morto, sem relação com MSW.

Corrigido com path relativo + handler responsivo ao cenário
(`mocks/handlers.ts`) e 3 atributos `data-testid`/`data-risk`/`role`
adicionados aos elementos já existentes de `sensor-monitor.tsx` (zero
mudança de layout/lógica) — ver README §4.14 (achado atualizado) ou o
commit `fix(frontend): corrige integração MSW/Playwright do Dashboard
(SSE + predictions)` para o detalhe completo. Resultado final:
`failure_alert.spec.ts` 9/9,
`dashboard_flow.spec.ts` 6/6, `maintenance_assistant.spec.ts` 7/7 (RNF-53
incluso), suíte completa 34/34, `--repeat-each=2` 44/44 sem intermitência.
Não verificado: se o runner real do GitHub Actions reproduzia o problema
original antes da correção (não era necessário verificar depois de
corrigido na origem).

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

## RF-27 / RNF-55 — Monitoramento de data drift (Evidently + Celery Beat)

- **PSI observado no ambiente de dev/demo atual é alto (~2.7, dominado por
  `Oil_temperature`)** — resultado REAL (Evidently rodou de verdade contra
  Postgres real, não um valor fabricado), mas provavelmente um artefato do
  simulador: o "current" (últimas 24h de `predictions`) reflete o replay
  SEQUENCIAL do simulador (`SensorSimulator`, RF-13) — uma janela temporal
  estreita e não-aleatória do parquet — contra um "reference" amostrado
  ALEATORIAMENTE do dataset inteiro (RF-27 §Fase 4). Em produção real, com
  operação variada ao longo do dia, esse efeito tende a ser menor; sem
  histórico de incidentes reais de drift para calibrar, `MIN_CURRENT_ROWS`
  (30) e o tamanho/seed da amostra de referência (5000/42) são escolhas de
  engenharia documentadas, não valores empiricamente validados.
- **`drift_detected=True` não aciona nenhuma notificação** (Telegram/
  e-mail) — deliberado, fora do escopo desta task ("não alterar... fluxo
  de notificações"). Hoje só é visível via `GET /monitoring/drift`. Se
  alertar humanos sobre drift se tornar um requisito, avaliar reaproveitar
  `CriticalFailureNotificationService`/Celery (RF-24/RNF-50) — nunca
  duplicar essa infraestrutura.
- **Celery Beat disparando autonomamente às 03:00 UTC não foi observado em
  tempo real** (exigiria esperar até esse horário) — validado por: (a)
  inspeção do `beat_schedule` real carregado pelo `celery-beat` (schedule
  diário, mesmo timezone UTC do RNF-50) e (b) disparo manual da MESMA task
  registrada (`daily_drift_analysis_task.delay()`) contra o
  `celery-worker`/Redis/Postgres reais — ponta a ponta idêntica ao que o
  Beat dispararia, só o gatilho (manual vs. cron) é diferente.
- **Reference é uma amostra FIXA do dataset de treinamento original** — não
  há mecanismo de atualização automática do baseline (ex.: após um
  re-treino legítimo do modelo com dados mais recentes). Se o baseline
  precisar mudar no futuro, é uma edição manual de
  `DriftMonitor.load_reference_data()`/`REFERENCE_SAMPLE_SEED`.
- **Sem UI de frontend** — deliberado, fora do escopo desta task ("Não
  criar UI para drift").
- **Distinção explícita**: RF-27 monitora **data drift** (distribuição das
  7 features de entrada analógicas — TP2, TP3, H1, DV_pressure,
  Reservoirs, Oil_temperature, Motor_current), nunca **degradação de
  performance do modelo** (que exigiria rótulos verdadeiros/ground truth
  de falhas reais, não coletados em produção) — ver README §4.15.

## RNF-60 / RNF-61 — DVC (Data Version Control)

- **Escopo do `dvc.yaml` = `ingest` → `train_random_forest` → `train_xgboost`,
  não o pipeline de ML inteiro. Reavaliado explicitamente (não apenas
  reafirmado) numa auditoria de acompanhamento** — decisão por script, não
  uma desculpa genérica de "são experimentais":
  - **`promote_model.py`: exclusão definitiva, motivo técnico, não de
    conveniência.** Seu input real é o estado *mutável* de runs já logados
    num servidor MLflow (`MlflowClient.search_runs(...order_by=[metric
    DESC], max_results=1)` — pega o "melhor run até agora"), não um
    conjunto de arquivos com hash estável. DVC detecta staleness por hash de
    dependência; não existe hash para "o melhor run que a MLflow tiver
    quando isto rodar". Declarar isso como stage seria reprodutibilidade de
    fachada — o resultado dependeria de quantos treinos alguém já rodou
    contra aquele MLflow, não do código/dado versionado.
  - **`train_mlp.py`: teria a MESMA classe de problema de determinismo já
    documentada para o Optuna, só que pior.** Auditoria desta rodada
    encontrou que o script **nunca chama `pl.seed_everything`** (nenhuma
    ocorrência de "seed" no arquivo) — diferente de `train_sequential.py`
    (linha ~342) e `train_autoencoder.py` (linha ~256), que semeiam
    corretamente. Registrar `train_mlp` como stage "reproduzível" no
    `dvc.yaml` hoje seria inventar uma garantia que o código não tem.
  - **`train_sequential.py` (TCN/BiLSTM/PatchTST) e `train_autoencoder.py`
    são, tecnicamente, os candidatos mais próximos de virarem stages** — são
    semeados (`pl.seed_everything`) e toleram MLflow indisponível (fallback
    gracioso, `logger=False`). Mesmo assim, decisão de MANTER fora nesta
    task por três motivos que não são "talvez": (a) nenhum é o
    `ACTIVE_MODEL` de produção (`random_forest_v2`, vencedor do benchmark
    RF-18/RNF-36 — `.env.example`); (b) treinam em dezenas de minutos a
    horas por arquitetura (vs. minutos do RF/XGBoost), o que inviabilizaria
    validar de verdade idempotência/rebuild/push/pull dentro desta task;
    (c) adicioná-los agora, só porque tecnicamente caberiam, seria escopo
    além do que foi pedido — a instrução desta auditoria foi explícita:
    "não adicione stages apenas por precaução". Se um retreino automatizado
    dos modelos sequenciais virar requisito futuro, são os candidatos
    naturais a novos stages (o padrão `cache: false` já estabelecido aqui
    se estende diretamente) — registrado aqui como trabalho futuro
    explícito, não implementado agora.
- **Optuna (`train_xgboost.py`) não é semeado (`sampler=TPESampler(seed=...)`
  ausente).** `random_state=42` cobre o split treino/teste, o K-Fold interno
  e o `XGBClassifier`, mas a escolha de quais hiperparâmetros o TPE testa em
  cada trial pode variar entre execuções "do zero" (sem
  `data/optuna/xgboost_study.db` prévio). Lacuna pré-existente do script, não
  introduzida por esta task — não corrigida aqui porque alterar a lógica de
  treino está fora do escopo de uma task de versionamento de dados. Na
  prática, isso só importa quando o `study.db` é apagado: com ele presente
  (caso normal de reprodução local), `load_if_exists=True` reaproveita os
  trials já rodados e o resultado é estável.
- **`data/optuna/xgboost_study.db` é deliberadamente excluído do
  `dvc.yaml`.** É um SQLite que ACUMULA trials entre execuções — não é uma
  função pura de seus deps declarados, então não se encaixa no contrato
  "mesmos deps ⇒ mesmo out" que o DVC espera. Continua gitignored/local,
  como já era antes desta task.
- **O stage `train_xgboost` usa `--n-trials 5`** (o "quick smoke run" já
  documentado no próprio script), não os `--n-trials 100` de um retreino de
  produção real — escolha deliberada para manter `dvc repro` rápido o
  bastante para validar de verdade (idempotência, rebuild, push/pull) dentro
  desta task. Um retreino de produção real continua sendo
  `python -m src.train_xgboost --n-trials 100`, rodado manualmente fora do
  `dvc repro` automático (o `dvc.yaml` não impede isso).
- **MinIO é uma instância de desenvolvimento/teste, não produção.** O
  projeto não tinha nenhum MinIO/S3 pré-existente (auditado antes de criar);
  a instância adicionada em `docker-compose.yml` é a mais simples possível
  (single-node, sem TLS, credenciais placeholder em `.env.example`). Para um
  ambiente real, trocar `endpointurl` do remote DVC (`apps/ml/.dvc/config`)
  para o S3/MinIO real e fornecer credenciais reais só via variável de
  ambiente/secret — nunca no `.dvc/config` versionado.
- **`data/processed/metropt3.parquet` e os artefatos de `models/` continuam
  commitados diretamente no Git** (`cache: false` no `dvc.yaml`), não
  migrados para o DVC. **Reavaliado explicitamente numa auditoria de
  acompanhamento** (a alternativa — migrar para DVC e remover do Git — foi
  considerada de novo, não só reafirmada):
  - Migrar para DVC exigiria que TODO consumidor desses arquivos rode
    `dvc pull` antes de usá-los — isso inclui o CI do backend
    (`test-python`, que hoje faz apenas `actions/checkout` e lê
    `metropt3.parquet` direto do working tree para ~34 testes, ver
    `.github/workflows/ci.yml`) e qualquer `docker compose up` de
    desenvolvimento que monta `apps/ml/models/` no backend. Isso
    acoplaria infraestrutura de CI/dev a um MinIO/S3 real — exatamente o
    tipo de dependência externa que o item de CI abaixo evita
    deliberadamente para o próprio `dvc.yaml`.
  - `models/*.onnx`/`*.joblib` são consumidos em produção pelo backend via
    bind-mount direto do checkout do Git (`ACTIVE_MODEL`, `model_registry.py`)
    — nunca passam por `dvc pull` em runtime. Removê-los do Git sem mudar
    esse mecanismo de carregamento quebraria o boot da API num checkout
    limpo; mudar o mecanismo de carregamento está fora do escopo de uma
    task de versionamento de dados (seria alteração de arquitetura de
    backend/deploy, explicitamente proibida nesta task).
  - Tamanho: o footprint binário em Git NÃO é trivial —
    `metropt3.parquet` ≈ 29 MB, `random_forest_final.joblib` ≈ 12,4 MB,
    `random_forest_v2.onnx` ≈ 6 MB, mais ~5 MB dos modelos de DL — total
    ~55 MB rastreados no Git, e cada retreino real commita um blob novo de
    ~47 MB no histórico. Isso NÃO é ideal e é o argumento mais forte a
    favor da migração eventual. Ainda assim, é ordens de magnitude abaixo
    do dataset bruto de ~208 MB (esse sim migrado para o DVC nesta task).
  - **Conclusão desta task (escopo limitado), com ressalva explícita**:
    parquet e modelos ficam no Git com `cache: false` (mesmo padrão
    RNF-56/57) porque migrá-los agora exigiria mexer no CI do backend e no
    bootstrap do compose — fora do escopo. O `dvc.yaml` já os rastreia
    como saídas (grafo/staleness) sem assumir a posse do armazenamento.
    **Follow-up recomendado** (não feito aqui, requer decisão do
    responsável): remover o `cache: false` de `metropt3.parquet` +
    `models/random_forest_*` + `models/xgboost_*`, deixar o DVC gerenciá-los
    de verdade (cache + remote), gitignorá-los, e adicionar um passo
    `dvc pull` ao job `test-python` do CI e ao bootstrap de
    desenvolvimento. Só aí o repositório atinge de fato o "sem binários
    grandes regeneráveis no Git" que a RNF-61 idealiza.
- **`train_random_forest` falhou 1× em 3 execuções reais** durante esta
  validação — crash do backend `loky` do joblib no meio do `GridSearchCV`
  (`n_jobs=2`), com `resource_tracker: leaked file/folder objects` e um
  `FileNotFoundError` na limpeza do memmap temporário do Windows, sem
  disco cheio, sem processo zumbi e sem lock preso. Reexecução simples
  (`dvc repro`, sem mudar nada) passou. É fragilidade conhecida do
  multiprocessing do joblib/loky no Windows sob cargas paralelas pesadas
  repetidas — NÃO é defeito do DVC, NÃO foi introduzida por esta task (o
  mesmo `train_random_forest.py` rodou OK 2× antes com config idêntica) e
  NÃO foi "corrigida" mexendo em `n_jobs`/GridSearch (seria alterar lógica
  de ML). Registrada como fragilidade operacional real: em máquina
  Windows, contar com `dvc repro` reexecutável na primeira tentativa.
- **Determinismo é por-artefato, não uniforme.** Comprovado empiricamente
  num ciclo completo apagar→`dvc pull`→retreino: `metropt3.parquet` e os
  `.joblib` dos modelos treinados saem **byte-idênticos** entre execuções
  (seeds funcionam). Já os `.onnx` saem com **bytes diferentes mas
  inferência idêntica ao bit** (diferença é ordem de serialização do
  protobuf, não o modelo — o `.joblib` de origem é idêntico e o export foi
  verificado: `max|Δ|` sklearn-vs-ONNX = 0). Os `*_card.json` diferem
  **só** no campo `trained_at` (timestamp ISO). Nenhuma dessas diferenças
  é defeito; documentadas para não alegar um determinismo byte-a-byte que
  o pipeline não tem no export ONNX.
- **`train_xgboost --n-trials 5` NÃO roda 5 trials se `data/optuna/`
  `xgboost_study.db` já existir** (o caso normal neste repo — o arquivo
  local tem 100 trials acumulados). Nesse caso o script loga
  `Study already has 100 trials — skipping optimisation` e reusa os
  melhores hiperparâmetros já encontrados. Só num ambiente 100% limpo
  (sem o `.db`, que é gitignored e não vai pro DVC) o `--n-trials 5`
  dispara 5 trials reais do zero — e aí, por causa do `TPESampler` não
  semeado (item acima), os hiperparâmetros escolhidos podem diferir. O
  comentário "quick smoke run" no `dvc.yaml` está correto para o caso
  limpo mas pode confundir; nuance registrada aqui.
- **Binários de checkpoint no Git (não relacionado a RNF-60/61 — só
  documentado).** `git ls-files` acusa `apps/ml/checkpoints/`
  `best-epoch=05-val_f1=0.8645.ckpt` (~628 KB) — que ESTÁ em `.gitignore`
  mas foi commitado antes da regra existir, então a regra nunca surtiu
  efeito nele — e `apps/ml/models/checkpoints/autoencoder_v1-*.ckpt` (2
  arquivos ~1,3 MB cada, um deles um duplicado `-v1` aparentemente
  acidental), esses sem nenhuma regra de ignore. São snapshots
  intermediários de treino do PyTorch Lightning, não artefatos de
  serving. Fora do escopo desta task (remover/untrackear arquivo exige
  decisão do responsável); registrado para avaliação futura —
  `git rm --cached` nesses 3 + regra de ignore, ou migração para DVC.
- **5 arquivos `*.onnx.data`** (pesos externos dos modelos de DL:
  mlp/tcn/bilstm/patchtst/autoencoder, ~2,3 MB somados) são rastreados
  pelo Git mas passam despercebidos num filtro ingênuo por extensão
  (`.onnx`) — não terminam em `.onnx`. Mesma categoria/decisão dos `.onnx`
  correspondentes (ficam no Git, consumidos por bind-mount pelo backend);
  citados aqui só para a auditoria de binários ficar completa.

## Pré-existente (não introduzido por nenhuma das tasks acima)

- `apps/backend/tests/test_simulator.py` falha na coleta com `IndexError: 3`
  — confirmado via `git stash` como pré-existente antes da RF-22. Contorno
  usado em toda validação: `pytest --ignore=tests/test_simulator.py`.
