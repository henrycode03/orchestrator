"""Apply the narrow C1 empty-tools compatibility patch to installed OpenClaw."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


EXPECTED_OPENCLAW_VERSION = "2026.4.10"
OPENCLAW_ROOT = Path("/usr/lib/node_modules/openclaw")
PATCHES = (
    (
        "node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js",
        "openai-completions",
    ),
    (
        "node_modules/@mariozechner/pi-ai/dist/providers/openai-responses.js",
        "openai-responses",
    ),
)
OLD_SNIPPET = "if (context.tools) {"
NEW_SNIPPET = "if (context.tools && context.tools.length > 0) {"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version() -> str:
    package = json.loads((OPENCLAW_ROOT / "package.json").read_text(encoding="utf-8"))
    return str(package.get("version", ""))


def _state() -> list[dict[str, object]]:
    version = _package_version()
    rows: list[dict[str, object]] = []
    for relative, name in PATCHES:
        path = OPENCLAW_ROOT / relative
        source = path.read_text(encoding="utf-8")
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": _sha256(path),
                "old_snippet_count": source.count(OLD_SNIPPET),
                "new_snippet_count": source.count(NEW_SNIPPET),
                "mode": oct(path.stat().st_mode & 0o777),
            }
        )
    return [{"openclaw_version": version}, *rows]


def _validate_state(state: list[dict[str, object]], *, patched: bool) -> None:
    version = state[0]["openclaw_version"]
    if version != EXPECTED_OPENCLAW_VERSION:
        raise RuntimeError(
            f"unsupported OpenClaw version {version!r}; expected "
            f"{EXPECTED_OPENCLAW_VERSION!r}"
        )
    for row in state[1:]:
        if not row["exists"]:
            raise RuntimeError(f"missing serializer: {row['path']}")
        expected = "new_snippet_count" if patched else "old_snippet_count"
        if row[expected] != 1:
            raise RuntimeError(
                f"unexpected {row['name']} state: {expected}={row[expected]}"
            )


def _apply() -> None:
    before = _state()
    _validate_state(before, patched=False)
    for relative, _ in PATCHES:
        path = OPENCLAW_ROOT / relative
        source = path.read_text(encoding="utf-8")
        path.write_text(source.replace(OLD_SNIPPET, NEW_SNIPPET), encoding="utf-8")
    after = _state()
    _validate_state(after, patched=True)
    print(json.dumps({"operation": "apply", "before": before, "after": after}))


def _restore() -> None:
    before = _state()
    _validate_state(before, patched=True)
    for relative, _ in PATCHES:
        path = OPENCLAW_ROOT / relative
        source = path.read_text(encoding="utf-8")
        path.write_text(source.replace(NEW_SNIPPET, OLD_SNIPPET), encoding="utf-8")
    after = _state()
    _validate_state(after, patched=False)
    print(json.dumps({"operation": "restore", "before": before, "after": after}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("check", "apply", "restore"))
    args = parser.parse_args()
    try:
        state = _state()
        if args.operation == "check":
            _validate_state(state, patched=True)
            print(json.dumps({"operation": "check", "state": state}))
        elif args.operation == "apply":
            _apply()
        else:
            _restore()
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"C1 patch {args.operation} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
