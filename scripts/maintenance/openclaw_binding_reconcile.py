#!/usr/bin/env python3
"""Inspect and reconcile ephemeral OpenClaw binding residue (Phase 37 Pre-B-F2B).

Report-only by default. ``--apply`` removes only STALE_CONFIRMED artifacts in
the managed binding root (lock free, metadata revalidated under the lock).
Legacy pre-F2B ``/tmp/orchestrator-openclaw-binding-*`` directories, unknown
ownership, invalid metadata and persistent OpenClaw state are never removed.
File contents (including copied credentials) are never read or printed.

    python3 scripts/maintenance/openclaw_binding_reconcile.py
    python3 scripts/maintenance/openclaw_binding_reconcile.py --include-legacy-tmp
    python3 scripts/maintenance/openclaw_binding_reconcile.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.orchestration.execution.binding_reconciliation import (  # noqa: E402
    inventory_legacy_tmp_bindings,
    reconcile_binding_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="Managed binding root. Default: the root new bindings are created in.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Remove STALE_CONFIRMED managed artifacts instead of only reporting.",
    )
    parser.add_argument(
        "--include-legacy-tmp",
        action="store_true",
        help="Also inventory (never remove) pre-F2B /tmp binding directories.",
    )
    args = parser.parse_args()

    payload = {
        "managed": reconcile_binding_artifacts(
            Path(args.root).expanduser() if args.root else None, apply=args.apply
        )
    }
    if args.include_legacy_tmp:
        payload["legacy_tmp"] = inventory_legacy_tmp_bindings()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
