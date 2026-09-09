"""
Pydantic schemas — RF-25 / RNF-49 (configuração global de alerta).

`alert_threshold` é validado em [0.5, 0.95] pelo próprio Pydantic (`Field`
com `ge`/`le`) — um valor fora do intervalo nunca chega ao service, o
FastAPI responde 422 automaticamente. O banco reforça o mesmo intervalo via
CHECK constraint (`models/alert_settings.py`) como segunda linha de defesa.

`alert_email` usa `EmailStr` (via `email-validator`, já dependência do
projeto) — mesma regra de "não inventar uma regex própria" do enunciado da
task. `email_enabled=true` sem `alert_email` é rejeitado por um
`model_validator` — o banco reforça a mesma regra via CHECK constraint.
"""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, model_validator

ALERT_THRESHOLD_MIN = 0.5
ALERT_THRESHOLD_MAX = 0.95


class AlertSettingsResponse(BaseModel):
    """Resposta de `GET /v1/settings/alerts` e `PUT /v1/settings/alerts`."""

    alert_threshold: float = Field(
        ge=ALERT_THRESHOLD_MIN,
        le=ALERT_THRESHOLD_MAX,
        description="Limiar efetivo de alerta crítico — usado por CriticalFailureNotificationService (RF-24).",
    )
    telegram_enabled: bool = Field(
        description="Se a notificação crítica também é enviada via Telegram."
    )
    email_enabled: bool = Field(
        description="Se a notificação crítica também é enviada por e-mail (Resend)."
    )
    alert_email: str | None = Field(
        default=None, description="Endereço de destino quando email_enabled=true."
    )


class AlertSettingsUpdateRequest(BaseModel):
    """Corpo de `PUT /v1/settings/alerts`."""

    alert_threshold: float = Field(
        ge=ALERT_THRESHOLD_MIN,
        le=ALERT_THRESHOLD_MAX,
        description="Novo limiar — deve satisfazer 0.5 <= alert_threshold <= 0.95.",
    )
    telegram_enabled: bool = Field(default=True)
    email_enabled: bool = Field(default=False)
    alert_email: EmailStr | None = Field(default=None)

    @model_validator(mode="after")
    def _require_email_when_enabled(self) -> "AlertSettingsUpdateRequest":
        if self.email_enabled and not self.alert_email:
            raise ValueError("alert_email é obrigatório quando email_enabled=true.")
        return self


class NotificationTestResponse(BaseModel):
    """Resposta de `POST /v1/settings/alerts/test`."""

    message: str
