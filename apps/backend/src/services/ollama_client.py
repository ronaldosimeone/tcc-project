"""
Cliente Ollama — RF-22 / RNF-46.

HTTP puro via `httpx` (já presente em requirements.txt para outros usos —
nenhuma dependência nova) contra `POST {OLLAMA_BASE_URL}/api/chat`, API real
do Ollama confirmada contra a instância local do ambiente desta task
(`{"model", "messages": [...], "stream": false}` -> `{"message": {"content": ...}}`).

Processamento 100% local (RNF-46): nenhuma chamada a OpenAI/Anthropic/Gemini
ou qualquer API externa — `OLLAMA_BASE_URL` aponta para o Ollama rodando no
host (`host.docker.internal`, Docker Desktop resolve nativamente, sem
`extra_hosts`) ou em outra máquina da rede interna, nunca para a internet
pública por padrão desta configuração.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from src.core.exceptions import OllamaResponseError, OllamaUnavailableError

log = structlog.get_logger(__name__)


class OllamaClient:
    """Cliente para `POST /api/chat` do Ollama — sem streaming (RF-22 pede um
    Markdown completo por resposta, não um endpoint incremental)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Chama o Ollama com System Prompt + prompt do usuário separados
        (RF-22 §13 — nunca concatenados numa única string, para que o
        conteúdo do manual — dentro de `user_prompt` — nunca seja
        confundido com instrução de sistema).

        Levanta `OllamaUnavailableError` (503) para falha de rede/timeout/
        modelo inexistente, ou `OllamaResponseError` (502) se a resposta não
        tiver conteúdo utilizável. Nunca propaga a URL interna do Ollama.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            log.warning("ollama_timeout", model=self._model)
            raise OllamaUnavailableError(
                "Tempo limite excedido ao gerar a sugestão de manutenção."
            ) from exc
        except httpx.HTTPError as exc:
            log.warning("ollama_connection_error", error=str(exc))
            raise OllamaUnavailableError(
                "Não foi possível conectar ao serviço de geração de sugestões."
            ) from exc

        if response.status_code == 404:
            log.warning("ollama_model_not_found", model=self._model)
            raise OllamaUnavailableError(
                f"Modelo '{self._model}' não está disponível no Ollama."
            )
        if response.status_code != 200:
            log.warning("ollama_bad_status", status_code=response.status_code)
            raise OllamaUnavailableError(
                "Serviço de geração de sugestões retornou um erro inesperado."
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            log.warning("ollama_invalid_json")
            raise OllamaResponseError(
                "Resposta do serviço de geração de sugestões não é válida."
            ) from exc

        content = data.get("message", {}).get("content")
        if not content or not isinstance(content, str) or not content.strip():
            log.warning("ollama_empty_content", data_keys=list(data.keys()))
            raise OllamaResponseError(
                "Serviço de geração de sugestões retornou conteúdo vazio."
            )

        return content
