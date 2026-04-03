"""Metadados de deploy para comparar o que roda na nuvem com o repositório local."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
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

_REPO_ROOT = Path(__file__).resolve().parent


def revisao_git_curta() -> str:
    """SHA curto do commit atual (local) ou variáveis comuns de CI / nuvem."""
    for key in (
        "DEPLOY_REVISION",
        "GIT_COMMIT",
        "GITHUB_SHA",
        "COMMIT_SHA",
        "VERCEL_GIT_COMMIT_SHA",
        "CF_PAGES_COMMIT_SHA",
    ):
        v = os.environ.get(key)
        if v:
            return v[:12] if len(v) > 12 else v
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "desconhecido"
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def rotulo_deploy() -> str:
    """Texto único para sidebar / debug: commit + dica de override via secrets."""
    sha = revisao_git_curta()
    if os.environ.get("DEPLOY_REVISION"):
        return f"Deploy: {sha} (DEPLOY_REVISION)"
    return f"Deploy: {sha}"


def data_referencia_para_prompt() -> str:
    """Data ‘hoje’ no fuso de Brasília para idade, tempo de empresa e durações (atualizada a cada pergunta)."""
    tz = ZoneInfo("America/Sao_Paulo")
    agora = datetime.now(tz)
    legivel = f"{agora.day} de {_MESES_PT[agora.month]} de {agora.year}"
    iso = agora.date().isoformat()
    return f"{legivel} (America/Sao_Paulo), calendário {iso}"
