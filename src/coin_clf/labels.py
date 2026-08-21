import json
from pathlib import Path


def save_labels(idx_to_name: dict[int, str], path: Path) -> None:
    payload = {str(idx): name for idx, name in idx_to_name.items()}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_labels(path: Path) -> dict[int, str]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Label file not found: {path}")
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}
