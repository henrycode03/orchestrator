"""Certification-only binding for the existing discovery capture argument."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator

from app.services.orchestration.planning import read_only_discovery
from app.services.orchestration.planning.discovery_contract_capture import (
    DiscoveryContractCapture,
)


class DiscoveryCaptureBinding:
    """Inject one existing capture path while delegating to production code."""

    def __init__(self, capture_path: str | Path) -> None:
        self.capture_path = Path(capture_path)
        self.original = read_only_discovery.run_discovery_stage
        self.incoming_capture_path: str | None = None
        self.injected = False

    def create_target(self) -> None:
        parent = self.capture_path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(parent, 0o777)
        if self.capture_path.exists():
            raise FileExistsError(f"capture target already exists: {self.capture_path}")
        # The production adapter updates this same artifact in finally-safe
        # boundary order. The initial document proves the target existed before
        # the lifecycle/provider dispatch began.
        capture = DiscoveryContractCapture(self.capture_path)
        capture._persist()  # noqa: SLF001 - certification target creation only
        os.chmod(self.capture_path, 0o666)

    def __enter__(self) -> "DiscoveryCaptureBinding":
        self.create_target()

        def bound_stage(*args: Any, **kwargs: Any) -> Any:
            self.incoming_capture_path = (
                str(kwargs.get("capture_path"))
                if kwargs.get("capture_path") is not None
                else None
            )
            if kwargs.get("capture_path") is None:
                kwargs["capture_path"] = self.capture_path
                self.injected = True
            return self.original(*args, **kwargs)

        read_only_discovery.run_discovery_stage = bound_stage
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        read_only_discovery.run_discovery_stage = self.original


@contextmanager
def bind_discovery_capture(
    capture_path: str | Path,
) -> Iterator[DiscoveryCaptureBinding]:
    """Create and bind one compact capture target for a certification run."""

    binding = DiscoveryCaptureBinding(capture_path)
    with binding:
        yield binding
