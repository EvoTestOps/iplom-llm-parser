import logging

import regex as re

logger = logging.getLogger(__name__)

SLOT_TYPE = {
    "OID": {"numeric_ok": False, "spans_ws": False},
    "LOI": {"numeric_ok": False, "spans_ws": False},
    "OBN": {"numeric_ok": False, "spans_ws": False},
    "TID": {"numeric_ok": False, "spans_ws": False},
    "SID": {"numeric_ok": True, "spans_ws": False},
    "TDA": {"numeric_ok": True, "spans_ws": True},
    "CRS": {"numeric_ok": True, "spans_ws": False},
    "OBA": {"numeric_ok": True, "spans_ws": False},
    "STC": {"numeric_ok": True, "spans_ws": False},
    "OTP": {"numeric_ok": False, "spans_ws": True},
}
_DEFAULT_SLOT = {"numeric_ok": False, "spans_ws": False}


def compile_template(
    template: str, slot_regexes: list[str] | None = None
) -> re.Pattern:
    parts = template.split("<*>")
    n = len(parts) - 1
    if slot_regexes is None:
        slot_regexes = [r"\S+"] * n

    pieces = [re.escape(parts[0])]
    for i in range(n):
        pieces.append(f"({slot_regexes[i]})")
        pieces.append(re.escape(parts[i + 1]))
    return re.compile("^" + "".join(pieces) + "$", re.DOTALL)


def _infer_one_slot(values: list[str], is_trailing: bool, type_info: dict) -> str:
    if not values:
        return r".+" if (is_trailing and type_info["spans_ws"]) else r"\S+"

    stripped = [v.strip() for v in values]
    has_ws = any(any(c.isspace() for c in v) for v in stripped)

    if has_ws or (type_info["spans_ws"] and is_trailing):
        return r".+" if is_trailing else r".+?"

    if type_info["numeric_ok"]:
        if all(v.isdigit() for v in stripped):
            return r"\d+"
        if all(re.fullmatch(r"[0-9a-fA-F]+", v) for v in stripped):
            return r"[0-9a-fA-F]+"

    return r"\S+"


def infer_slot_regexes(
    template: str, samples: list[str], slot_types: list[str]
) -> list[str]:
    parts = template.split("<*>")
    n = len(parts) - 1
    if n == 0:
        return []

    boot = re.compile("^" + "(.+?)".join(re.escape(p) for p in parts) + "$", re.DOTALL)
    captures: list[list[str]] = [[] for _ in range(n)]
    for s in samples:
        try:
            m = boot.fullmatch(s, timeout=1.0)
        except TimeoutError:
            logger.warning("infer_slot_regexes: Regex match timeout exceeded")
            continue
        if not m:
            continue
        for i, g in enumerate(m.groups()):
            captures[i].append(g)

    out = []
    for i in range(n):
        t = slot_types[i] if i < len(slot_types) else ""
        type_info = SLOT_TYPE.get(t, _DEFAULT_SLOT)
        is_trailing = i == n - 1 and parts[-1] == ""
        out.append(_infer_one_slot(captures[i], is_trailing, type_info))
    return out
