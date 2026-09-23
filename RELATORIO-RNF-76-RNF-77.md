# RELATORIO-RNF-76-RNF-77.md

## RNF-76 — `GET /metrics` (observabilidade) / RNF-77 — Alerta visual de taxa de erro

---

## 1. Auditoria (Fase 1)

Antes de qualquer alteração, mapeado:

- **Backend**: `src/main.py` (create_app/lifespan), `src/routers/*` (health, predict, monitoring — este último é "Drift Monitoring", RF-27, sem relação com Prometheus), `src/services/model_service.py` (ModelService.predict/predict_from_features/predict_batch), `src/services/inference_pipeline.py` (loop contínuo de inferência, 1 execução por leitura SSE), `src/services/model_registry.py` (KNOWN_MODELS, 9 nomes fixos), `src/core/config.py`, `src/core/exceptions.py`, `Dockerfile`, `docker-compose.yml`.
- **Frontend**: `components/connection-status.tsx` (padrão visual de 3 estados já existente — verde/âmbar/vermelho, usado para SSE/WS), `components/sensor-monitor/header.tsx`, `hooks/*`, `lib/api-client.ts`.
- **Infraestrutura**: `grep` no repositório inteiro por `prometheus|grafana|/metrics` — **nenhum resultado** fora desta task. Nenhuma infraestrutura de observabilidade duplicada.

Nenhum `/metrics`, Prometheus, Grafana, `prometheus.yml` ou dashboard pré-existia.

---

## 2. RNF-76 — `GET /metrics`

### Endpoint

`GET /metrics` no FastAPI (`src/core/metrics.py::setup_http_instrumentation`), também alcançável em `GET /api/metrics` através do nginx — **mesmo modelo de exposição do resto da API** (o bloco `location /api/` já é um proxy genérico para toda a aplicação; não há um mecanismo de auth por rota além do que já existe em `/models`). Ver §7 "Segurança" para a avaliação explícita desta escolha.

### Biblioteca

`prometheus-fastapi-instrumentator==8.1.0` + `prometheus-client==0.26.0` — versões reais confirmadas via `pip index versions` no ambiente desta task (não assumidas da memória de treinamento, que estaria desatualizada), API introspectada diretamente do pacote instalado antes de escrever qualquer código de integração.

### Achado real durante a implementação — registry compartilhado quebra testes

A suíte de testes deste projeto chama `create_app()` repetidas vezes no mesmo processo (cada arquivo de teste cria sua própria app; `from src.main import create_app` já executa `app = create_app()` uma vez ao nível de módulo antes disso). Com o `CollectorRegistry` global padrão do `prometheus_client`, a **segunda** chamada a `metrics.default()` detecta "métrica já registrada" e devolve `None` silenciosamente (comportamento documentado da própria lib) — a segunda app em diante ficava com um middleware que existia mas nunca gravava nada (`instrumentations=[]`, verificado por inspeção direta do objeto).

**Correção**: `setup_http_instrumentation()` cria um `CollectorRegistry()` **novo a cada chamada** (não o global). As métricas de ML (módulo `src/core/metrics.py`, singletons criados uma única vez por processo) são reanexadas a este registry via `registry.register(collector)` para aparecerem juntas em `/metrics`. Verificado com duas chamadas consecutivas de `create_app()` no mesmo processo — cada uma conta corretamente seu próprio tráfego, isoladas uma da outra. Em produção (`create_app()` roda uma única vez por processo) o comportamento é idêntico ao registry global — nenhuma mudança de comportamento real, só correção de um bug de reentrância que só se manifestava sob teste.

### Métricas padrão (HTTP)

Via `prometheus-fastapi-instrumentator` (`metrics.default()`):

| Métrica | Labels | Tipo |
|---|---|---|
| `http_requests_total` | `method`, `status`, `handler` | Counter |
| `http_request_duration_seconds` | `method`, `handler` | Histogram (poucos buckets) |
| `http_request_duration_highr_seconds` | (nenhum) | Histogram (muitos buckets, usado para p95/p50) |
| `http_request_size_bytes` / `http_response_size_bytes` | `handler` | Summary |

Configuração: `should_group_status_codes=False` (mantém código exato — 200/422/429/500/503 — necessário para RNF-77 distinguir 4xx de 5xx), `should_ignore_untemplated=True` (requisição sem rota casada nunca vira métrica — protege contra cardinalidade não controlada de bots/probes), `excluded_handlers=["/metrics"]` (o próprio scrape não se autoconta).

`handler` usa o **template** da rota (ex. `/predict/`), nunca a URL resolvida — confirmado por inspeção direta do `/metrics` real (ver §6).

### Métricas customizadas de ML (`src/core/metrics.py`)

| Métrica | Labels | Utilidade operacional |
|---|---|---|
| `predictiq_inference_total` | `model`, `status` | Cobre POST /predict **e** o pipeline contínuo (`InferencePipelineService`, 1 execução/leitura SSE) — o único jeito de ver erro/volume de inferência do pipeline em background, que nunca passa por uma rota HTTP |
| `predictiq_inference_duration_seconds` | `model`, `status` | Latência real da inferência (predict_proba isolado), não confundida com latência HTTP total |
| `predictiq_predictions_total` | `model`, `prediction_class` (0/1) | Distribuição de classe prevista — taxa de "alarme" do modelo sem consultar o Postgres |
| `predictiq_batch_requests_total` | `model`, `status` | POST /predict/batch é 1 chamada vetorizada, não N inferências — contador próprio |
| `predictiq_batch_samples_total` | `model` | Amostras processadas com sucesso via batch |

Instrumentadas em `ModelService.predict_from_features` (cobre `predict()` E o pipeline contínuo, que convergem nesse mesmo método) e `ModelService.predict_batch` — dois pontos de instrumentação, sem duplicar lógica entre os 3 call-sites reais. `model_name` é agora um parâmetro de `ModelService.__init__`, passado por **todos** os 5 pontos de construção em `load_model_by_name`/`_load_sequential_model`/`_load_autoencoder_model` — validado por teste de integração parametrizado sobre os 9 modelos reais (`test_model_service_real_artifacts.py`).

### Cardinalidade dos labels (regras 16/17)

- `model`: 9 valores fixos (`ModelRegistry.KNOWN_MODELS`)
- `status`: `"success"` / `"error"` (2 valores)
- `prediction_class`: `"0"` / `"1"` (classificador binário)
- `handler` (HTTP): template de rota, nunca URL resolvida
- `status` (HTTP): código exato, mas só os que a API realmente devolve (conjunto pequeno e fixo)

Nenhum UUID de usuário, ID individual de equipamento, request ID aleatório, URL completa com parâmetros ou timestamp como label.

### Prometheus scrape — confirmado real

`docker compose up` (api, prometheus, cadvisor, grafana) → `GET /api/v1/targets` do Prometheus real:

```
cadvisor       http://cadvisor:8080/metrics   up
predictiq-api  http://api:8000/metrics        up
prometheus     http://localhost:9090/metrics  up
```

### Exemplo real (trecho de `/metrics` após tráfego real gerado nesta task)

```
predictiq_inference_total{model="random_forest_v2",status="success"} 585.0
predictiq_predictions_total{model="random_forest_v2",prediction_class="0"} 585.0
predictiq_batch_samples_total{model="random_forest_v2"} 2.0
http_requests_total{handler="/health",method="GET",status="200"} 5.0
http_requests_total{handler="/predict/",method="POST",status="200"} 1.0
http_requests_total{handler="/predict/",method="POST",status="422"} 1.0
http_requests_total{handler="/predict/batch",method="POST",status="200"} 1.0
```

O valor 585 de `predictiq_inference_total` (vs. só 1 chamada manual a `POST /predict/`) confirma que a métrica captura o **pipeline contínuo** de verdade, rodando em background desde o startup do container — não é um número fabricado.

### Resultado RNF-76: **PASS**

Não apenas "`/metrics` retorna 200" (regra 8) — confirmado: métricas reais → Prometheus faz scrape real → PromQL real devolve os números corretos → Grafana consulta o Prometheus real e renderiza (ver §4).

---

## 3. RNF-77 — Alerta visual de taxa de erro

### Definição da taxa de erro (Fase 4)

```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
```

**Deliberadamente HTTP 5xx, não 4xx**: 4xx (422 payload inválido, 429 rate limit do RNF-19) é o cliente "errando", não a aplicação falhando — misturar os dois inflaria a métrica com tráfego de rate-limit saudável e mascararia problemas reais do servidor. O dashboard Grafana expõe 4xx e 5xx **separadamente** (dois painéis distintos), nunca misturados.

Erro de inferência (`predictiq_inference_total{status="error"}`) é um sinal **deliberadamente separado**: nem toda inferência passa por HTTP (pipeline contínuo), então não entra na conta de taxa de erro HTTP — evita a confusão explícita que a Fase 4 pede para não cometer.

### Thresholds (documentados, não arbitrários)

| Nível | Condição |
|---|---|
| NORMAL | `error_rate < 1%` |
| WARNING | `1% ≤ error_rate < 5%` |
| CRITICAL | `error_rate ≥ 5%` |

Racional: banda de alerta precoce comum em prática de SRE — um SLO de 99% de disponibilidade deixa 1% de "orçamento de erro"; o limiar WARNING em 1% dá margem de resposta antes desse orçamento se esgotar. Configuráveis via `.env` (`ERROR_RATE_WARNING_THRESHOLD`/`ERROR_RATE_CRITICAL_THRESHOLD`), não hardcoded — decisão operacional, não constante de código.

### Fonte dos dados (Fase 9)

`GET /observability/error-rate` — endpoint **interno** do backend (`src/services/observability_service.py`) que consulta o Prometheus via sua HTTP API (`/api/v1/query`, rede Docker interna) e devolve um JSON pequeno e já agregado:

```json
{
  "status": "NORMAL",
  "error_rate": 0.0,
  "window": "5m",
  "threshold_warning": 0.01,
  "threshold_critical": 0.05,
  "prometheus_reachable": true
}
```

**Por que não o frontend falar direto com o Prometheus**: o Prometheus não tem autenticação própria por padrão — expô-lo ao browser deixaria qualquer PromQL arbitrário acessível publicamente. Este endpoint expõe só o resultado de UMA query fixa, nunca repassa PromQL do cliente. Nunca levanta exceção — falha de rede/parsing degrada para `status="NORMAL"`, `prometheus_reachable=false` (nunca um alerta falso a partir de um dado ausente; o widget nunca derruba o Dashboard).

### Componente visual — reaproveitado, não criado do zero (Fase 8)

Localizado o padrão já existente: `components/connection-status.tsx` (RF-17) já tinha exatamente 3 estados visuais (verde/âmbar/vermelho, "Online/Reconectando/Offline") para os canais SSE e WebSocket. Extendido com um 3º pill opcional ("API"), reaproveitando a MESMA lógica de cor/dot, com rótulos próprios ("Normal/Atenção/Crítico" em vez de "Online/Reconectando/Offline" — evita confundir taxa de erro com estado de conectividade). Nenhum componente novo criado para o indicador em si.

`hooks/use-error-rate.ts` — poll a cada 15s (mesmo padrão do `use-sensor-data.ts`, intervalo maior porque a janela PromQL já é de 5 minutos: consultar a cada 5s não traria dado mais fresco, só carga extra — RNF-77 Fase 9 "baixa carga"). Erros de rede silenciosos, mantém o último status conhecido.

### Comportamento de recuperação

Verificado via testes determinísticos (`tests/test_observability_service.py`, backend): `NORMAL → WARNING → CRITICAL` e de volta para `NORMAL` quando a taxa cai — a classificação (`_classify()`) é uma função pura testada em cada fronteira (`>=`, não `>`), e o hook do frontend (`__tests__/use-error-rate.test.ts`) confirma que o `ConnectionStatus` reflete a mudança em ambas direções (teste "de CRITICAL para NORMAL, o rótulo atualiza").

### Verificação end-to-end real (Fase 10)

Cadeia completa confirmada com dados reais, não mockados, via `docker compose up` real:

```
POST /predict/ real (200) ──┐
POST /predict/ inválido (422) ──┤→ http_requests_total real
POST /predict/batch real (200) ─┘
        ↓
   GET /metrics real (confirma os contadores)
        ↓
   Prometheus scrape real (target UP, confirmado via /api/v1/targets)
        ↓
   PromQL real (/api/v1/query, resultado correto)
        ↓
   GET /observability/error-rate real → {"prometheus_reachable": true, ...}
        ↓
   GET http://localhost/api/observability/error-rate via nginx (200, rede real)
        ↓
   Dashboard React real — pill "API: Normal" renderizado no navegador (screenshot capturado)
```

**Nota sobre threshold-crossing ao vivo**: não foi provocado um 5xx real sustentado na stack ao vivo para observar CRITICAL em produção (rule 3 do RNF-76/77 pede para não simular requisições artificialmente para "produzir dados" — gerar uma falha real e sustentada exigiria quebrar deliberadamente um componente da stack por vários minutos). A transição de estado NORMAL→WARNING→CRITICAL→NORMAL em si é verificada por testes determinísticos que exercitam o código de classificação real (`_classify()`) contra respostas reais do formato Prometheus — consistente com a própria orientação da Fase 12 ("evitar testes baseados só em snapshot visual; testar o comportamento do componente"). A cadeia de *infraestrutura* (API→métrica→Prometheus→backend→frontend) foi 100% verificada com tráfego real.

### Resultado RNF-77: **PASS**

Não apenas "existe um componente visual estático" (regra 9) — o pill reage a dados reais computados a partir do Prometheus real, a fonte de dados é testável e reproduzível, e a transição de estado (incluindo recuperação) é coberta por teste real do código de classificação.

---

## 4. Grafana

- **URL local**: `http://localhost:3001` (não publicado como 3000 — já usado pelo `frontend`)
- **Login**: `GRAFANA_ADMIN_USER`/`GRAFANA_ADMIN_PASSWORD` via `.env` (default de dev `admin`/`changeme-grafana-admin`, mesmo padrão do `MINIO_ROOT_PASSWORD` já existente no projeto — trocar em produção)
- **Datasource**: Prometheus, provisionado automaticamente (`infra/grafana/provisioning/datasources/prometheus.yml`) — confirmado via `GET /api/datasources` real e via proxy-query real (`up` → 3 jobs, todos UP)
- **Dashboard**: `infra/grafana/dashboards/predictiq-overview.json` — **autoral**, não copiado de um ID público do grafana.com (não havia "origem" a registrar); 14 painéis reais + 3 separadores de seção, provisionado automaticamente (`infra/grafana/provisioning/dashboards/dashboards.yml`, `allowUiUpdates: false` — dashboard é código versionado)
- **Queries principais**: listadas em §2/§3 acima (requests/sec, taxa de erro 4xx/5xx, latência p95/p50, CPU/RAM, `up`, inferências/erros/predições/batch por modelo)

Todas as queries **validadas manualmente contra o Prometheus real** desta task (screenshots capturados) antes de serem salvas no JSON — nenhuma copiada sem checar se a métrica/label realmente existe.

### Achado real — cAdvisor não resolve nomes de container neste host (Docker Desktop/WSL2)

Ambiente desta task: Docker Desktop no Windows sobre WSL2 (`docker version` → `linux/6.6.87.2-microsoft-standard-WSL2`). Ao inspecionar `container_cpu_usage_seconds_total` direto do cAdvisor (`GET /metrics` do próprio cAdvisor), só aparecem 4 séries de cgroup de **topo** (`id="/"`,`"/docker"`,`"/docker/buildkit"`,`"/restricted"`) — **nenhuma série por container individual**, nenhum label `name=`. Tentativa de correção (montagens padrão + `privileged: true` + mount extra de `/dev/disk`) não resolveu — confirmado ser uma limitação conhecida do cAdvisor em Docker Desktop (virtualização aninhada: o cAdvisor lê `/sys`/`/var/lib/docker` esperando um host Linux nativo, que não é o caso aqui). Revertido o `privileged: true` (pior postura de segurança, sem benefício).

**Mitigação**: os painéis de CPU/RAM têm uma 2ª série (`{id="/docker"}`, agregado de TODOS os containers Docker) além da série por-container — a série agregada **tem dado real** neste ambiente (confirmado: ~0,15 core, ~3,25 GiB, ambos reais e crescendo). Em um host Linux nativo (alvo real de produção deste tipo de stack), a série por-container funcionaria sem alteração nenhuma — o `docker-compose.yml`/`prometheus.yml` não precisam de nenhuma mudança para isso.

**Resultado**: "disponibilidade dos containers" — **PASS** (painel `up`, 100% real). CPU/RAM — **PASS parcial neste ambiente de desenvolvimento** (agregado real, não por-container; a causa é documentada, não mascarada, e a configuração está correta para o alvo de produção real).

---

## 5. Docker

| Serviço | Exposição | Depende de |
|---|---|---|
| `prometheus` | interna (`expose: 9090`, nunca `ports:`) | `api`, `cadvisor` |
| `cadvisor` | interna (`expose: 8080`) | — |
| `grafana` | publicada (`ports: 3001:3000`) — única exceção, é UI para humano | `prometheus` |

Nenhum serviço pré-existente duplicado ou alterado além do necessário (`api` só ganhou as 2 dependências novas no `requirements.txt`).

---

## 6. Segurança (Fase 14)

- **Grafana sem senha hardcoded**: credenciais via `GF_SECURITY_ADMIN_USER`/`GF_SECURITY_ADMIN_PASSWORD`, lidas de `.env` — nunca no `docker-compose.yml`. Default de dev claramente marcado como tal (mesmo padrão do `MINIO_ROOT_PASSWORD` já existente).
- **Prometheus sem exposição desnecessária**: só `expose:`, nunca `ports:` — inacessível fora da rede Docker.
- **`/metrics` — avaliação explícita da exposição** (regra da Fase 14): `GET /api/metrics` é alcançável externamente através do nginx, pelo MESMO motivo que qualquer outra rota da API é (`location /api/` é um proxy genérico para toda a aplicação — não há autenticação por rota fora de `/models`, que já usa `X-Admin-Token`). Avaliado como **aceitável para esta arquitetura**: o conteúdo exposto é só contadores/histogramas agregados (ver §6.1 abaixo), sem PII/tokens/payloads — o mesmo risco relativo de qualquer outro endpoint público já existente no projeto (ex. `/health`). Uma futura camada de auth em `/metrics` (ex. IP allowlist no nginx) é uma melhoria possível, não implementada nesta task por não ter sido pedida e por manter consistência com o modelo de exposição já adotado pelo resto do projeto.
- **cAdvisor**: montagens somente-leitura (`:ro`) do socket Docker/`/sys`/`/var/lib/docker` — sem acesso de escrita; exposto só internamente.
- **Ausência de secrets/PII/payloads/tokens nas métricas** (verificado por inspeção direta de `/metrics` real, ver §2): só nomes de modelo (conjunto fixo de 9), status (`success`/`error`), classe prevista (`0`/`1`), método/status/rota HTTP. Nenhum token, nenhuma leitura de sensor, nenhum ID de equipamento, nenhum dado de usuário.

---

## 7. Regressão (Fase 15)

### Backend

| Check | Resultado |
|---|---|
| `pytest` (exclui 2 arquivos que exigem Redis com porta publicada no host — pré-existente, não relacionado a esta task) | **762 passed, 1 xfailed** (xfail pré-existente, `xgboost_v1.joblib` sem `feature_names_in_`, documentado em PENDENCIAS.md antes desta task) |
| `pytest` (suíte completa, incluindo os 2 arquivos acima) | 9 erros de conexão Redis (`localhost:6379` recusado) — **pré-existente**, `docker-compose.yml::redis` nunca publicou porta no host (`expose`, não `ports`, por design — ver comentário no próprio compose); não relacionado a nenhuma mudança desta task |
| coverage | **95.45%** (piso: 85%) — `src/core/metrics.py`, `src/schemas/observability.py`, `src/services/observability_service.py`, `src/routers/observability.py`: **100%** cada |
| ruff | limpo |
| black | limpo (após `black` nos arquivos novos/modificados) |
| mypy (`--ignore-missing-imports`, mesma invocação do CI) | `Success: no issues found in 118 source files` |
| import-linter | `Contracts: 1 kept, 0 broken` |
| mutation (mutmut, escopo `src/services/model_service.py` — único arquivo já coberto pelo `paths_to_mutate` do projeto que esta task modificou) | **235/257 mortos (91,4%)** — ver detalhamento abaixo |

### Frontend

| Check | Resultado |
|---|---|
| `pnpm test` | **330 passed**, 23 falhas — **idênticas, mesmos 5 arquivos**, às 23 falhas pré-existentes já confirmadas via `git stash` na task RNF-74/75 anterior (não relacionadas: lógica pura de `getRiskLevel`/FIFO de alertas, sem relação com observabilidade) |
| Regressão introduzida e corrigida durante esta task | `sensor-monitor.test.tsx::"chama fetch GET /predictions a cada intervalo de polling"` contava TODAS as chamadas a `fetch` — o novo `useErrorRateStatus` adiciona uma chamada independente a `/observability/error-rate`, quebrando a contagem exata. Corrigido filtrando por URL (`/api/v1/predictions`) em vez de contar todas as chamadas — o teste volta a validar exatamente o que seu nome promete, sem depender de chamadas não relacionadas. Confirmado: suíte retorna aos 23 falhas/316 passes pré-existentes + as novas 14 passam. |
| ESLint | limpo (1 warning pré-existente não relacionado, `test-airleak.js`) |
| `tsc --noEmit` | limpo |

Nenhum threshold de qualidade reduzido. Nenhuma regra de negócio alterada/removida.

### Detalhamento da mutação (`model_service.py`)

Primeira execução (257 mutantes): 231 mortos, **26 sobreviventes**. Cada um dos 26 inspecionado individualmente (`mutmut show <id>`):

- **4 sobreviventes reais em código NOVO desta task** — corrigidos com testes adicionais, confirmados mortos numa segunda rodada:
  - `model_name: str = "unknown"` (default de `load_model()`) — string nunca era verificada; teste real (`test_load_model_falls_back_to_alternate_filename`) já chamava `load_model()` sem `model_name` mas não tinha assert sobre o label resultante — adicionado.
  - `time.perf_counter() - start` → `+ start` (2 ocorrências, sucesso e erro de `record_inference`) — meus testes originais só checavam "duração > 0"/"contador aumentou", o que um `+` (que também dá um número positivo, só que enorme) não derrubava. Corrigido com limite superior explícito (`0 < delta < 1.0`) — só a subtração produz um delta pequeno e plausível para uma chamada instantânea.
  - `failure_probability >= self._threshold` → `>` dentro de `predict_batch` — o teste de fronteira exato (`>=` no limiar) já existia para `predict_from_features` (task RNF-64/65) mas nunca foi replicado para o loop de `predict_batch` (lógica duplicada, adicionada na RNF-72/73) — adicionado o equivalente para batch.
- **17 sobreviventes pré-existentes, código não tocado por esta task** — mutações de string em mensagens de log/exceção (`logger.info("[RF-10]...")` → `logger.info("XX[RF-10]...XX")`) em linhas que já existiam antes de RNF-76/77; nenhum teste (desta ou de tasks anteriores) afirma sobre o conteúdo exato dessas strings de log. Não corrigido — fora do escopo desta task (não são linhas que esta task introduziu ou modificou), e corrigir retroativamente todo o arquivo não foi pedido.
- **5 mutantes equivalentes (matematicamente impossíveis de matar)** — trocas de `|` por `&` em anotações de tipo de variável local (`card: dict[str, Any] | None = ...`). Com `from __future__ import annotations` (já ativo no arquivo antes desta task), anotações de variável local nunca são avaliadas em runtime pelo CPython — o bytecode gerado é IDÊNTICO com `|` ou `&`. Confirmado por inspeção: nenhum teste, por mais exaustivo que fosse, poderia diferenciar os dois.

Resultado final: **235/257 mortos (91,4%)**. Os 22 sobreviventes restantes são 17 pré-existentes (fora do escopo desta task) + 5 equivalentes (impossíveis de matar) — **zero sobreviventes em lógica nova introduzida por RNF-76/77**.

---

## 8. Arquivos alterados

### Backend
- **Novos**: `src/core/metrics.py`, `src/routers/observability.py`, `src/schemas/observability.py`, `src/services/observability_service.py`, `tests/test_metrics.py`, `tests/test_observability_service.py`, `tests/test_observability_endpoint.py`
- **Modificados**: `src/main.py` (wiring do instrumentator + router), `src/services/model_service.py` (parâmetro `model_name`, instrumentação de `predict_from_features`/`predict_batch`), `src/core/config.py` (5 novas settings), `requirements.txt` (2 dependências), `tests/test_model_service_unit.py` (+13 testes — 11 da instrumentação inicial + 2 para fechar sobreviventes reais de mutação, ver §7), `tests/test_model_service_real_artifacts.py` (+1 assert por modelo + 1 assert extra fechando sobrevivente de mutação)

### Frontend
- **Novos**: `hooks/use-error-rate.ts`, `__tests__/use-error-rate.test.ts`, `__tests__/connection-status.test.tsx`
- **Modificados**: `lib/api-client.ts` (tipo + função `getErrorRateStatus`), `components/connection-status.tsx` (3º pill opcional), `components/sensor-monitor/header.tsx`, `components/sensor-monitor.tsx` (wiring do hook), `__tests__/sensor-monitor.test.tsx` (fix da regressão descrita acima)

### Infraestrutura
- **Novos**: `infra/prometheus/prometheus.yml`, `infra/grafana/provisioning/datasources/prometheus.yml`, `infra/grafana/provisioning/dashboards/dashboards.yml`, `infra/grafana/dashboards/predictiq-overview.json`
- **Modificados**: `docker-compose.yml` (+3 serviços: prometheus, cadvisor, grafana), `.env.example` (+6 variáveis)

---

## 9. Dependências adicionadas

- `prometheus-fastapi-instrumentator==8.1.0`
- `prometheus-client==0.26.0` (já vinha transitivo; fixado explicitamente)
- Imagens Docker novas: `prom/prometheus:v3.0.1`, `gcr.io/cadvisor/cadvisor:v0.49.1`, `grafana/grafana:11.4.0`

Nenhuma dependência removida.

---

## 10. Problemas encontrados (honestos, não mascarados)

1. **Registry Prometheus compartilhado quebrava testes** (§2) — corrigido com registry por-chamada.
2. **cAdvisor não resolve nomes de container em Docker Desktop/WSL2** (§4) — limitação de ambiente documentada, mitigada com série agregada; configuração correta para produção Linux nativa, sem mudança necessária.
3. **Regressão introduzida no teste de polling do frontend** (§7) — detectada e corrigida na mesma task, teste voltou a validar exatamente sua intenção original.
4. `/metrics` publicamente alcançável via nginx (mesmo modelo do resto da API) — avaliado e aceito, documentado explicitamente (§6), não é um "problema" no sentido de bug, mas uma decisão de exposição que a Fase 14 pede para registrar.
