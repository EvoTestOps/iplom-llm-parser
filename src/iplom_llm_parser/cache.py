from dataclasses import dataclass, field

import regex as re

from iplom_llm_parser.regex_comp import compile_template


@dataclass
class CacheEntry:
    template: str
    regex: re.Pattern
    slot_types: list[str] = field(default_factory=list)
    match_frequency: int = 0


class TemplateCache:
    def __init__(self):
        self._entries: list[CacheEntry] = []

    def insert(
        self,
        template: str,
        slot_regexes: list[str] | None,
        slot_types: list[str],
    ) -> CacheEntry:
        entry = CacheEntry(
            template=template,
            regex=compile_template(template, slot_regexes),
            slot_types=slot_types,
        )
        self._entries.append(entry)
        return entry

    def increment(self, entry: CacheEntry, count: int = 1) -> None:
        entry.match_frequency += count

    def remove(self, entry: CacheEntry) -> None:
        for i, e in enumerate(self._entries):
            if e is entry:
                self._entries.pop(i)
                return

    def match_message(self, message: str) -> tuple[CacheEntry, re.Match] | None:
        best_entry = None
        best_match = None
        best_key = None

        for entry in self._entries:
            m = entry.regex.fullmatch(message)
            if m is not None:
                cost = sum(len(g) for g in m.groups())
                key = (cost, -len(entry.template), -entry.match_frequency)
                if best_key is None or key < best_key:
                    best_key = key
                    best_entry = entry
                    best_match = m

        if best_entry is not None:
            return best_entry, best_match
        return None

    def __str__(self) -> str:
        lines = ["TemplateCache", f"  {len(self._entries)} templates\n"]
        for e in sorted(self._entries, key=lambda e: -e.match_frequency):
            lines.append(f"  [{e.match_frequency:>3}] {e.template}")
        return "\n".join(lines)
