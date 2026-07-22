"""Read/write oracle trace files used for perfect-expert prefetch baselines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class OracleTrace:
    model: str = ""
    token_count: int = 0
    active_experts: int = 0
    prompt_token_ids: List[int] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    prompt_text: str = ""
    generated_text: str = ""
    expert_trace: List[List[List[int]]] = field(default_factory=list)


def _parse_id_csv(line: str) -> List[int]:
    line = line.strip()
    if not line:
        return []
    return [int(x.strip()) for x in line.split(",") if x.strip()]


def _section(content: str, header: str) -> Optional[str]:
    start = content.find(header)
    if start == -1:
        return None
    start += len(header)
    end = content.find("\n================================================================================", start)
    if end == -1:
        end = len(content)
    return content[start:end].strip("\n")


def parse_oracle_trace(content: str) -> OracleTrace:
    trace = OracleTrace()

    m = re.search(r"^MODEL:\s*(.+)$", content, re.MULTILINE)
    if m:
        trace.model = m.group(1).strip()

    m = re.search(r"^TOKEN_COUNT:\s*(\d+)$", content, re.MULTILINE)
    if m:
        trace.token_count = int(m.group(1))

    m = re.search(r"^ACTIVE_EXPERTS:\s*(\d+)$", content, re.MULTILINE)
    if m:
        trace.active_experts = int(m.group(1))

    prompt_ids_section = _section(content, "PROMPT TOKEN IDS:\n")
    if prompt_ids_section is not None:
        trace.prompt_token_ids = _parse_id_csv(prompt_ids_section.splitlines()[0])

    gen_ids_section = _section(content, "GENERATED TOKEN IDS:\n")
    if gen_ids_section is not None:
        trace.generated_token_ids = _parse_id_csv(gen_ids_section.splitlines()[0])

    prompt_text = _section(content, "PROMPT TEXT:\n")
    if prompt_text is not None:
        trace.prompt_text = prompt_text

    generated_text = _section(content, "GENERATED TEXT:\n")
    if generated_text is not None:
        trace.generated_text = generated_text

    in_trace = False
    for line in content.splitlines():
        if "EXPERT TRACE" in line:
            in_trace = True
            continue
        if not in_trace:
            continue
        if not line.startswith("Token"):
            continue
        colon = line.find(":")
        if colon == -1:
            continue
        experts_str = line[colon + 1 :]
        layers: List[List[int]] = []
        for match in re.finditer(r"\[([^\]]*)\]", experts_str):
            layer_experts = _parse_id_csv(match.group(1))
            layers.append(layer_experts)
        if layers:
            trace.expert_trace.append(layers)

    if trace.token_count <= 0 and trace.expert_trace:
        trace.token_count = len(trace.expert_trace)
    return trace


def load_oracle_trace(path: str | Path) -> OracleTrace:
    text = Path(path).read_text(encoding="utf-8")
    return parse_oracle_trace(text)


def format_id_csv(ids: List[int]) -> str:
    return ",".join(str(i) for i in ids)
