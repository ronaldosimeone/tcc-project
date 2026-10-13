"""
Templates HTML dos e-mails do PredictIQ — RF-25 / RNF-49.

Funções puras (sem I/O): recebem os dados do alerta e devolvem o documento
HTML completo. O envio fica em `EmailNotificationAdapter`; o corpo em texto
simples continua sendo enviado junto (multipart) como fallback para clientes
que bloqueiam HTML.

Decisões de design
------------------
* Layout 100 % em `<table>` com estilos inline — é o único subconjunto de
  HTML/CSS que Gmail, Outlook e Apple Mail renderizam de forma consistente.
* Largura máxima de 600 px, fluida abaixo disso (mobile).
* A barra de probabilidade é uma tabela de duas células (não SVG/CSS grid),
  porque Gmail remove SVG e ignora `width: calc()`.
* Tema escuro via `prefers-color-scheme` (Apple Mail / Outlook.com / Gmail
  app); onde não houver suporte, o tema claro continua legível.
* TODO valor dinâmico passa por `html.escape` — o corpo agora é HTML, então
  um nome de equipamento com markup não pode virar injeção no e-mail.

Escala de risco: mesma do Dashboard (`lib/risk-thresholds.ts`) —
NORMAL < 0,30 ≤ ALERTA < 0,65 ≤ CRÍTICO.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from src.core.time_format import format_local_timestamp

_FONT = (
    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
    "'Helvetica Neue',Arial,sans-serif"
)
_MONO = "'SFMono-Regular',Menlo,Consolas,'Liberation Mono',monospace"

_ALERT_THRESHOLD = 0.30
_CRITICAL_THRESHOLD = 0.65

# Media query isolada (string simples, não f-string) para não escapar chaves.
_DARK_MODE_CSS = """
@media (prefers-color-scheme: dark) {
  .bg-page { background-color: #05080f !important; }
  .bg-card { background-color: #111a2b !important; }
  .bg-soft { background-color: #17233a !important; }
  .t-main { color: #f1f5f9 !important; }
  .t-muted { color: #94a3b8 !important; }
  .b-line { border-color: #24324d !important; }
}
@media only screen and (max-width: 620px) {
  .container { width: 100% !important; }
  .px { padding-left: 20px !important; padding-right: 20px !important; }
  .big { font-size: 56px !important; }
}
"""


@dataclass(frozen=True)
class _Risk:
    label: str
    color: str
    soft: str
    message: str


def _risk_for(probability: float) -> _Risk:
    if probability >= _CRITICAL_THRESHOLD:
        return _Risk(
            "CRÍTICO",
            "#dc2626",
            "#fee2e2",
            "Intervenção imediata recomendada.",
        )
    if probability >= _ALERT_THRESHOLD:
        return _Risk(
            "ALERTA",
            "#d97706",
            "#fef3c7",
            "Programe uma inspeção o quanto antes.",
        )
    return _Risk("NORMAL", "#059669", "#d1fae5", "Operação dentro do esperado.")


def _safe_url(url: str) -> str:
    """Só http(s) vira link — qualquer outro esquema cai para `#`."""
    candidate = url.strip()
    if candidate.lower().startswith(("http://", "https://")):
        return escape(candidate, quote=True)
    return "#"


def _button(*, label: str, url: str, color: str) -> str:
    # Botão "à prova de Outlook": <a> dentro de <td bgcolor>, sem imagens.
    return f"""
<table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center">
  <tr>
    <td align="center" bgcolor="{color}" style="border-radius:12px;">
      <a href="{url}" target="_blank"
         style="display:inline-block;padding:15px 34px;font-family:{_FONT};
                font-size:15px;font-weight:700;line-height:20px;color:#ffffff;
                text-decoration:none;border-radius:12px;letter-spacing:0.2px;">
        {escape(label)} &rarr;
      </a>
    </td>
  </tr>
</table>"""


def _detail_row(label: str, value_html: str, *, last: bool = False) -> str:
    border = "" if last else "border-bottom:1px solid #e2e8f0;"
    return f"""
<tr>
  <td class="b-line t-muted" width="38%" valign="top"
      style="padding:13px 0;{border}font-family:{_FONT};font-size:13px;
             color:#64748b;">{escape(label)}</td>
  <td class="b-line t-main" valign="top" align="right"
      style="padding:13px 0;{border}font-family:{_FONT};font-size:14px;
             font-weight:600;color:#0f172a;">{value_html}</td>
</tr>"""


def _step(number: int, title: str, text: str) -> str:
    return f"""
<tr>
  <td width="40" valign="top" style="padding:0 0 16px 0;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0">
      <tr><td align="center" width="28" height="28" bgcolor="#2563eb"
              style="border-radius:14px;font-family:{_FONT};font-size:13px;
                     font-weight:700;color:#ffffff;">{number}</td></tr>
    </table>
  </td>
  <td valign="top" style="padding:0 0 16px 0;">
    <div class="t-main" style="font-family:{_FONT};font-size:14px;
         font-weight:700;color:#0f172a;">{escape(title)}</div>
    <div class="t-muted" style="font-family:{_FONT};font-size:13px;
         line-height:19px;color:#64748b;padding-top:2px;">{escape(text)}</div>
  </td>
</tr>"""


def _shell(*, title: str, preheader: str, hero_html: str, body_html: str) -> str:
    """Moldura comum: cabeçalho de marca + hero + corpo + rodapé."""
    return f"""<!DOCTYPE html>
<html lang="pt-BR" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <meta name="supported-color-schemes" content="light dark">
  <title>{escape(title)}</title>
  <style>{_DARK_MODE_CSS}</style>
</head>
<body class="bg-page" style="margin:0;padding:0;background-color:#eef2f7;
      -webkit-text-size-adjust:100%;">
  <!-- Pré-visualização na caixa de entrada -->
  <div style="display:none;max-height:0;overflow:hidden;opacity:0;
              color:transparent;mso-hide:all;">{escape(preheader)}</div>

  <table role="presentation" class="bg-page" width="100%" cellspacing="0"
         cellpadding="0" border="0" bgcolor="#eef2f7">
    <tr>
      <td align="center" style="padding:32px 12px;">

        <table role="presentation" class="container" width="600" cellspacing="0"
               cellpadding="0" border="0" style="width:600px;max-width:600px;">

          <!-- Marca -->
          <tr>
            <td class="px" style="padding:0 8px 18px 8px;">
              <table role="presentation" width="100%" cellspacing="0"
                     cellpadding="0" border="0">
                <tr>
                  <td style="font-family:{_FONT};font-size:20px;font-weight:800;
                             letter-spacing:-0.3px;color:#0f172a;" class="t-main">
                    <span style="color:#2563eb;">&#9679;</span>&nbsp;PredictIQ
                  </td>
                  <td align="right" class="t-muted"
                      style="font-family:{_FONT};font-size:11px;font-weight:600;
                             letter-spacing:1.4px;color:#64748b;
                             text-transform:uppercase;">
                    Manutenção Preditiva
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Cartão principal -->
          <tr>
            <td class="bg-card" bgcolor="#ffffff"
                style="border-radius:20px;overflow:hidden;
                       box-shadow:0 10px 30px rgba(15,23,42,0.10);">
              <table role="presentation" width="100%" cellspacing="0"
                     cellpadding="0" border="0">
                {hero_html}
                {body_html}
              </table>
            </td>
          </tr>

          <!-- Rodapé -->
          <tr>
            <td class="px" align="center"
                style="padding:26px 24px 8px 24px;font-family:{_FONT};
                       font-size:12px;line-height:19px;color:#94a3b8;">
              <div class="t-muted" style="color:#64748b;font-weight:600;">
                PredictIQ &middot; Plataforma de manutenção preditiva
                &middot; MetroPT-3
              </div>
              <div style="padding-top:6px;">
                Você recebe este e-mail porque o canal de e-mail está ativo em
                <em>Configurações &rarr; Alertas</em>. Para ajustar o limite ou
                desativar, acesse o painel.
              </div>
              <div style="padding-top:10px;color:#cbd5e1;">
                Projeto acadêmico &middot; TCC &middot; Indústria 4.0
              </div>
            </td>
          </tr>
        </table>

      </td>
    </tr>
  </table>
</body>
</html>"""


def _hero(*, risk: _Risk, chip: str, headline: str, subline: str) -> str:
    return f"""
<tr>
  <td bgcolor="#0b1220"
      style="background-color:#0b1220;
             background-image:linear-gradient(135deg,#0b1220 0%,#16213a 55%,{risk.color} 160%);
             padding:34px 36px 30px 36px;" class="px">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0">
      <tr>
        <td bgcolor="{risk.color}"
            style="border-radius:999px;padding:6px 14px;font-family:{_FONT};
                   font-size:11px;font-weight:800;letter-spacing:1.6px;
                   color:#ffffff;text-transform:uppercase;">
          &#9679;&nbsp;{escape(chip)}
        </td>
      </tr>
    </table>
    <div style="font-family:{_FONT};font-size:27px;line-height:34px;
                font-weight:800;color:#ffffff;padding-top:16px;
                letter-spacing:-0.4px;">{headline}</div>
    <div style="font-family:{_FONT};font-size:14px;line-height:21px;
                color:#cbd5e1;padding-top:8px;">{subline}</div>
  </td>
</tr>"""


def render_critical_failure_email_html(
    *,
    equipment_name: str,
    probability: float,
    timestamp: str,
    dashboard_url: str,
) -> str:
    """E-mail de falha crítica — hero vermelho, medidor, detalhes e CTA."""
    risk = _risk_for(probability)
    pct = max(0, min(100, round(probability * 100)))
    rest = 100 - pct
    equipment = escape(equipment_name)
    url = _safe_url(dashboard_url)

    hero = _hero(
        risk=risk,
        chip="Falha crítica detectada",
        headline=f"{equipment} precisa de atenção agora",
        subline=(
            "O modelo preditivo identificou um padrão compatível com falha "
            "iminente neste equipamento."
        ),
    )

    # Barra de probabilidade: duas células com largura em %.
    rest_cell = (
        f'<td width="{rest}%" bgcolor="#e2e8f0" style="font-size:0;'
        f'line-height:0;height:12px;border-radius:0 6px 6px 0;">&nbsp;</td>'
        if rest > 0
        else ""
    )
    fill_radius = "6px" if rest == 0 else "6px 0 0 6px"
    bar = f"""
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
  <tr>
    <td width="{max(pct, 1)}%" bgcolor="{risk.color}"
        style="font-size:0;line-height:0;height:12px;border-radius:{fill_radius};">&nbsp;</td>
    {rest_cell}
  </tr>
</table>"""

    body = f"""
<tr>
  <td class="px bg-card" bgcolor="#ffffff" style="padding:32px 36px 8px 36px;">

    <!-- Medidor -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0" class="bg-soft" bgcolor="#f8fafc"
           style="border-radius:16px;border:1px solid #e2e8f0;">
      <tr>
        <td class="b-line" style="padding:24px 26px 22px 26px;">
          <div class="t-muted" style="font-family:{_FONT};font-size:11px;
               font-weight:700;letter-spacing:1.5px;color:#64748b;
               text-transform:uppercase;">Probabilidade de falha</div>
          <table role="presentation" width="100%" cellspacing="0"
                 cellpadding="0" border="0">
            <tr>
              <td valign="bottom" style="padding-top:6px;">
                <span class="big" style="font-family:{_FONT};font-size:68px;
                      line-height:68px;font-weight:800;color:{risk.color};
                      letter-spacing:-2px;">{pct}</span><span
                      style="font-family:{_FONT};font-size:28px;font-weight:700;
                      color:{risk.color};">%</span>
              </td>
              <td valign="bottom" align="right" style="padding-bottom:10px;">
                <span style="display:inline-block;background-color:{risk.soft};
                      color:{risk.color};font-family:{_FONT};font-size:12px;
                      font-weight:800;letter-spacing:1.2px;padding:6px 12px;
                      border-radius:999px;">{risk.label}</span>
              </td>
            </tr>
          </table>
          <div style="padding-top:14px;">{bar}</div>
          <table role="presentation" width="100%" cellspacing="0"
                 cellpadding="0" border="0">
            <tr>
              <td class="t-muted" style="padding-top:8px;font-family:{_FONT};
                  font-size:11px;color:#94a3b8;">0%</td>
              <td class="t-muted" align="center" style="padding-top:8px;
                  font-family:{_FONT};font-size:11px;color:#94a3b8;">
                alerta 30%</td>
              <td class="t-muted" align="center" style="padding-top:8px;
                  font-family:{_FONT};font-size:11px;color:#94a3b8;">
                crítico 65%</td>
              <td class="t-muted" align="right" style="padding-top:8px;
                  font-family:{_FONT};font-size:11px;color:#94a3b8;">100%</td>
            </tr>
          </table>
        </td>
      </tr>
    </table>

    <!-- Detalhes -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0" style="padding-top:22px;">
      {_detail_row("Equipamento", f'<span style="font-family:{_MONO};">{equipment}</span>')}
      {_detail_row("Nível de risco", f'<span style="color:{risk.color};">{risk.label}</span> &mdash; {escape(risk.message)}')}
      {_detail_row("Detectado em", escape(format_local_timestamp(timestamp)))}
      {_detail_row("Origem", "Modelo preditivo &middot; MetroPT-3", last=True)}
    </table>

    <!-- CTA -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0">
      <tr><td align="center" style="padding:30px 0 6px 0;">
        {_button(label="Abrir o Dashboard", url=url, color=risk.color)}
      </td></tr>
      <tr><td align="center" class="t-muted" style="padding:10px 0 4px 0;
            font-family:{_FONT};font-size:12px;color:#94a3b8;">
        Ou copie o endereço: <span style="font-family:{_MONO};
        color:#64748b;">{escape(dashboard_url)}</span>
      </td></tr>
    </table>
  </td>
</tr>

<tr>
  <td class="px bg-card" bgcolor="#ffffff" style="padding:26px 36px 34px 36px;">
    <div style="border-top:1px solid #e2e8f0;" class="b-line">&nbsp;</div>
    <div class="t-main" style="font-family:{_FONT};font-size:15px;
         font-weight:800;color:#0f172a;padding:2px 0 16px 0;">
      O que fazer agora</div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0">
      {_step(1, "Confirme no Dashboard", "Abra o monitor do equipamento e verifique a tendência das pressões TP2/TP3, da corrente do motor e da temperatura do óleo.")}
      {_step(2, "Consulte o Assistente de IA", "Gere um plano de manutenção fundamentado nos manuais técnicos do equipamento.")}
      {_step(3, "Reconheça o alerta", "Depois de agir, reconheça o alerta no painel para manter o histórico consistente.")}
    </table>
  </td>
</tr>"""

    return _shell(
        title="PredictIQ — Falha crítica detectada",
        preheader=(
            f"{equipment_name}: probabilidade de falha de {pct}% — "
            "intervenção imediata recomendada."
        ),
        hero_html=hero,
        body_html=body,
    )


def render_test_email_html(*, dashboard_url: str = "") -> str:
    """E-mail de teste — hero verde, claramente marcado como simulação."""
    risk = _risk_for(0.0)
    hero = _hero(
        risk=risk,
        chip="Teste de notificação",
        headline="Tudo certo: o canal de e-mail está funcionando",
        subline=(
            "Esta mensagem confirma que o PredictIQ consegue entregar alertas "
            "na sua caixa de entrada."
        ),
    )
    button = (
        f'<tr><td align="center" style="padding:26px 0 6px 0;">'
        f'{_button(label="Abrir o Dashboard", url=_safe_url(dashboard_url), color="#2563eb")}'
        f"</td></tr>"
        if dashboard_url
        else ""
    )
    body = f"""
<tr>
  <td class="px bg-card" bgcolor="#ffffff" style="padding:32px 36px 34px 36px;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0" class="bg-soft" bgcolor="#f8fafc"
           style="border-radius:16px;border:1px solid #e2e8f0;">
      <tr><td style="padding:22px 26px;">
        <div class="t-main" style="font-family:{_FONT};font-size:15px;
             font-weight:800;color:#0f172a;">Configuração validada</div>
        <div class="t-muted" style="font-family:{_FONT};font-size:13px;
             line-height:20px;color:#64748b;padding-top:6px;">
          A chave do provedor, o remetente e o endereço de destino estão
          corretos. Quando uma probabilidade de falha ultrapassar o limite
          configurado, você receberá um e-mail como este &mdash; com o
          equipamento, o nível de risco e um atalho para o painel.
        </div>
      </td></tr>
    </table>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0" style="padding-top:18px;">
      {_detail_row("Tipo", "Mensagem de teste")}
      {_detail_row("Impacto", "Nenhum &mdash; não há falha real de equipamento", last=True)}
    </table>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           border="0">{button}</table>
  </td>
</tr>"""
    return _shell(
        title="PredictIQ — Teste de Notificação",
        preheader="Teste concluído: o canal de e-mail do PredictIQ está ativo.",
        hero_html=hero,
        body_html=body,
    )
