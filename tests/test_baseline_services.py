from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.heartbeat.service import HeartbeatService
from nanobot.runtime.state import state


async def test_heartbeat_service_skips_when_suspended(tmp_path):
    calls = 0

    async def on_heartbeat(_: str) -> str:
        nonlocal calls
        calls += 1
        return "ok"

    service = HeartbeatService(workspace=tmp_path, on_heartbeat=on_heartbeat)
    (tmp_path / "HEARTBEAT.md").write_text("do thing", encoding="utf-8")

    state.register_suspended_service("heartbeat")
    try:
        await service._tick()
    finally:
        state.unregister_suspended_service("heartbeat")

    assert calls == 0


async def test_cron_service_skips_jobs_when_suspended(tmp_path):
    calls = 0

    async def on_job(_job) -> str:
        nonlocal calls
        calls += 1
        return "ok"

    service = CronService(store_path=tmp_path / "jobs.json", on_job=on_job)
    job = service.add_job(
        name="test",
        schedule=CronSchedule(kind="every", every_ms=1),
        message="run",
    )
    job.state.next_run_at_ms = 0

    state.register_suspended_service("cron")
    try:
        await service._on_timer()
    finally:
        state.unregister_suspended_service("cron")

    assert calls == 0
