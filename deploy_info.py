"""Data de referência para o prompt do assistente (fuso de Brasília)."""
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
    """Data ‘hoje’ no fuso de Brasília para idade, tempo de empresa e durações (atualizada a cada pergunta)."""
    tz = ZoneInfo("America/Sao_Paulo")
    agora = datetime.now(tz)
    legivel = f"{agora.day} de {_MESES_PT[agora.month]} de {agora.year}"
    iso = agora.date().isoformat()
    return f"{legivel} (America/Sao_Paulo), calendário {iso}"
