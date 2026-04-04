"""
Utilitário de data para o prompt do sistema (fuso America/Sao_Paulo).

Usado pelo assistente para calcular idade, tempo de empresa e intervalos com
uma referência de “hoje” consistente com o calendário brasileiro.

Versão do pacote da aplicação: 1.0
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_MESES_PT = (
    "",
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def data_referencia_para_prompt() -> str:
    """
    Retorna a data corrente em linguagem natural + ISO, fuso de Brasília.

    Atualizada a cada invocação (cada pergunta no chat), para o modelo não
    desatualizar durações ao longo da sessão.
    """
    tz = ZoneInfo("America/Sao_Paulo")
    agora = datetime.now(tz)
    legivel = f"{agora.day} de {_MESES_PT[agora.month]} de {agora.year}"
    iso = agora.date().isoformat()
    return f"{legivel} (America/Sao_Paulo), calendário {iso}"
