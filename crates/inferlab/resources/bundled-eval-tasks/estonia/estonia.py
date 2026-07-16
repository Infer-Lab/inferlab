from __future__ import annotations

import json
import re
import string
from pathlib import Path

_ROOT = Path(__file__).parent
_FINAL_PREFIX = re.compile(
    r"(?is)^(?:(?:final\s+)?answer|(?:therefore|thus),?\s+the\s+answer\s+is|"
    r"the\s+answer\s+is|the\s+country\s+is)\s*[:\-]?\s*"
)


def load_dataset(**_: object) -> object:
    import datasets  # type: ignore[import-not-found]

    metadata = json.loads((_ROOT / "dataset.json").read_text(encoding="utf-8"))
    row = {
        "sample_id": metadata["sample_id"],
        "target": metadata["target"],
        "prompt": (_ROOT / "prompt.txt").read_text(encoding="utf-8"),
    }
    return datasets.DatasetDict({"test": datasets.Dataset.from_list([row])})


def _terminal_answer(text: object) -> tuple[str | None, str]:
    if not isinstance(text, str):
        return None, "missing"
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), "post_think"
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip(), "final_line"
    return "", "final_line"


def _normalize(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = _FINAL_PREFIX.sub("", text.strip())
    normalized = stripped.strip(string.whitespace + string.punctuation)
    normalized = re.sub(r"\s+", " ", normalized).casefold()
    return normalized or None


def process_results(doc: dict[str, object], results: list[str]) -> dict[str, float]:
    response = results[0] if len(results) == 1 else None
    answer, source = _terminal_answer(response)
    normalized = _normalize(answer)
    expected = _normalize(str(doc["target"]))
    outcome = (
        "unparseable" if normalized is None else "passed" if normalized == expected else "wrong"
    )
    doc["_inferlab_task_evidence"] = {
        "terminal_answer": answer,
        "terminal_answer_source": source,
        "normalized_terminal_answer": normalized,
        "expected_normalized_answer": expected,
        "classified_outcome": outcome,
    }
    return {"estonia_pass": 1.0 if outcome == "passed" else 0.0}
