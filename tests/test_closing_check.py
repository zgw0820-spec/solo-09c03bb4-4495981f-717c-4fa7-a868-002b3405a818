"""Deterministic tests for the batch closing configuration verification.

These exercise ``Coordinator._run_batch`` with a scripted service client, so the
exact interleaving (a direct fault edit landing after a service's probe has
answered) can be reproduced without timing sleeps.
"""
from __future__ import annotations

import asyncio
import uuid

from app.config import ROLES, SERVICES, Settings
from app.coordinator import Coordinator
from app.store import Store


def _evidence(version: int, epoch: int, endpoint: str) -> dict:
    return {
        "kind": "ok",
        "ok": True,
        "http_status": 200,
        "latency_ms": 1.0,
        "timed_out": False,
        "unreachable": False,
        "reason": "alive" if endpoint == "live" else "ready",
        "detail": None,
        "state_version": version,
        "config_epoch": epoch,
        "local_ready_ok": None if endpoint == "live" else True,
        "blocking_dependencies": [],
        "timed_out_dependencies": [],
    }


def _probes(version: int, epoch: int) -> dict:
    return {
        service: {
            "service": service,
            "role": ROLES[service],
            "live": _evidence(version, epoch, "live"),
            "ready": _evidence(version, epoch, "ready"),
            "live_ok": True,
            "ready_ok": True,
            "failed": False,
            "failure_kind": None,
            "state_version": version,
            "config_epoch": epoch,
            "blocking_dependencies": [],
            "timed_out_dependencies": [],
            "local_ready_ok": True,
        }
        for service in SERVICES
    }


def _states(*, epoch: int, unreachable: tuple[str, ...] = (), error: str = "down") -> list[dict]:
    return [
        {
            "service": service,
            "role": ROLES[service],
            "reachable": service not in unreachable,
            "config_epoch": None if service in unreachable else epoch,
            "state_version": 1,
            "error": error if service in unreachable else None,
        }
        for service in SERVICES
    ]


class ScriptedClient:
    """Just enough of ``ServiceClient`` for one batch's call sequence."""

    def __init__(self, snapshots: list[list[dict]], probes: dict) -> None:
        self._snapshots = list(snapshots)
        self._probes = probes
        self.client = self  # ``Coordinator._close_client`` calls client.client.aclose()

    async def snapshot_states(self) -> list[dict]:
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    async def probe_all(self) -> dict:
        return self._probes

    async def aclose(self) -> None:
        pass


def _run_batch(tmp_path, fake: ScriptedClient) -> tuple[dict, Coordinator, str]:
    async def scenario():
        settings = Settings(database_path=str(tmp_path / "diag.db"))
        store = Store(settings.database_path)
        coordinator = Coordinator(settings, store)
        batch_id = uuid.uuid4().hex
        seq = coordinator._next_seq()
        expected_version = coordinator.state_version
        store.create_running_batch(batch_id, seq, expected_version)

        async def make_client() -> ScriptedClient:
            return fake

        coordinator._make_client = make_client  # type: ignore[method-assign]
        await coordinator._run_batch(batch_id, seq, expected_version)
        return coordinator.get_batch(batch_id), coordinator, batch_id

    record, coordinator, batch_id = asyncio.run(scenario())
    return record, coordinator, batch_id


def test_closing_epoch_change_forces_retest_even_when_probe_evidence_agrees(tmp_path):
    # The probe answered under epoch 1 with the batch's state version; a direct
    # edit arrived afterwards, before the batch closed.
    fake = ScriptedClient(
        snapshots=[_states(epoch=1), _states(epoch=2)],
        probes=_probes(version=1, epoch=1),
    )
    record, coordinator, _ = _run_batch(tmp_path, fake)
    result = record["result"]

    assert record["status"] == "completed"
    assert record["outcome"] == "needs_retest"
    assert result["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    assert result["primary_root_cause"] is None
    assert result["independent_faults_count"] == 0
    assert result["affected_services"] == []
    joined = " ".join(result["retest_reasons"])
    assert "configuration changed during this batch" in joined
    assert "auth" in joined
    # The stale probe evidence really did look consistent: that is the race.
    assert all(
        probe["state_version"] == 1 and probe["config_epoch"] == 1
        for probe in result["probes"].values()
    )
    # Verification must not consume the change: the watcher still owns the
    # epoch bookkeeping and has to bump the version afterwards.
    assert coordinator.snapshot_epochs() == {service: None for service in SERVICES}


def test_closing_verification_failure_forces_retest(tmp_path):
    # Every service answered during probing, but auth's admin state cannot be
    # re-read when the batch closes: the configuration cannot be verified.
    fake = ScriptedClient(
        snapshots=[_states(epoch=1), _states(epoch=1, unreachable=("auth",))],
        probes=_probes(version=1, epoch=1),
    )
    record, coordinator, _ = _run_batch(tmp_path, fake)
    result = record["result"]

    assert record["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    joined = " ".join(result["retest_reasons"])
    assert "auth: could not verify configuration after probing" in joined
    assert "direct change during this batch cannot be ruled out" in joined
    # A service that could not be verified is never turned into a root cause.
    assert all(cause["service"] != "auth" for cause in result["root_causes"])


def test_untouched_batch_still_reports_healthy(tmp_path):
    fake = ScriptedClient(
        snapshots=[_states(epoch=1), _states(epoch=1)],
        probes=_probes(version=1, epoch=1),
    )
    record, _, _ = _run_batch(tmp_path, fake)
    assert record["status"] == "completed"
    assert record["outcome"] == "healthy"
    assert record["result"]["retest_reasons"] == []


def test_direct_change_not_yet_stamped_into_version_forces_retest(tmp_path):
    # Probing started before the watcher noticed the direct change: baseline and
    # closing reads agree, but the epoch no longer matches what the batch's
    # state version is known to describe.
    fake = ScriptedClient(
        snapshots=[_states(epoch=2), _states(epoch=2)],
        probes=_probes(version=1, epoch=2),
    )

    async def scenario():
        settings = Settings(database_path=str(tmp_path / "diag2.db"))
        store = Store(settings.database_path)
        coordinator = Coordinator(settings, store)
        coordinator._expected_epochs = {service: 1 for service in SERVICES}
        batch_id = uuid.uuid4().hex
        seq = coordinator._next_seq()
        version = coordinator.state_version
        store.create_running_batch(batch_id, seq, version)

        async def make_client():
            return fake

        coordinator._make_client = make_client  # type: ignore[method-assign]
        await coordinator._run_batch(batch_id, seq, version)
        return coordinator.get_batch(batch_id), coordinator

    record, coordinator = asyncio.run(scenario())
    assert record["outcome"] == "needs_retest"
    assert record["result"]["root_causes"] == []
    assert any(
        "direct change has not been stamped" in reason
        for reason in record["result"]["retest_reasons"]
    )
    # Still not consumed by the batch.
    assert coordinator.snapshot_epochs() == {service: 1 for service in SERVICES}
