"""Tests for the trust REST API endpoints."""

import json
import threading
import time
import urllib.request
import urllib.error

from nomotic.api import NomoticAPIServer
from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.monitor import DriftConfig
from nomotic.registry import ArchetypeRegistry, OrganizationRegistry, ZoneValidator
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.store import MemoryCertificateStore
from nomotic.types import Action, AgentContext, TrustProfile


def _setup_server():
    """Create a running API server with a runtime for testing."""
    sk, _vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(issuer_id="test", signing_key=sk, store=store)

    drift_cfg = DriftConfig(window_size=10, check_interval=3, min_observations=5)
    runtime = GovernanceRuntime(
        config=RuntimeConfig(enable_fingerprints=True, drift_config=drift_cfg),
    )
    runtime.set_certificate_authority(ca)

    server = NomoticAPIServer(
        ca,
        archetype_registry=ArchetypeRegistry.with_defaults(),
        zone_validator=ZoneValidator(),
        org_registry=OrganizationRegistry(),
        runtime=runtime,
        host="127.0.0.1",
        port=0,
    )
    httpd = server._build_server()
    httpd.server_address  # ensure bound
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    return httpd, runtime, port


def _get(port: int, path: str) -> tuple[int, dict]:
    """GET request helper."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _evaluate(runtime: GovernanceRuntime, agent_id: str, action_type: str, target: str):
    """Run a single evaluation through the runtime."""
    action = Action(agent_id=agent_id, action_type=action_type, target=target)
    ctx = AgentContext(agent_id=agent_id, trust_profile=TrustProfile(agent_id=agent_id))
    return runtime.evaluate(action, ctx)


class TestTrustAPI:
    def test_trust_report_404_unknown_agent(self):
        httpd, runtime, port = _setup_server()
        try:
            status, data = _get(port, "/v1/trust/unknown-agent")
            assert status == 404
        finally:
            httpd.shutdown()

    def test_trust_report_200_after_evaluation(self):
        httpd, runtime, port = _setup_server()
        try:
            # Evaluate some actions first
            for _ in range(5):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a")
            assert status == 200
            assert data["agent_id"] == "a"
            assert "current_trust" in data
            assert "successful_actions" in data
            assert "violation_count" in data
            assert "violation_rate" in data
            assert "trajectory" in data
        finally:
            httpd.shutdown()

    def test_trust_report_includes_trajectory_summary(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(5):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a")
            assert status == 200

            traj = data["trajectory"]
            assert "total_events" in traj
            assert "net_delta" in traj
            assert "trend" in traj
            assert "sources" in traj
            assert "recent_events" in traj
        finally:
            httpd.shutdown()

    def test_trajectory_endpoint_returns_events(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(5):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a/trajectory")
            assert status == 200
            assert data["agent_id"] == "a"
            assert "events" in data
            assert "total" in data
            assert len(data["events"]) == data["total"]
        finally:
            httpd.shutdown()

    def test_trajectory_filter_since(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(3):
                _evaluate(runtime, "a", "read", "/data")

            cutoff = time.time()
            time.sleep(0.02)

            for _ in range(3):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, f"/v1/trust/a/trajectory?since={cutoff}")
            assert status == 200
            # Should only have events after cutoff
            assert data["total"] <= 3
        finally:
            httpd.shutdown()

    def test_trajectory_filter_source(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(5):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a/trajectory?source=verdict")
            assert status == 200
            for event in data["events"]:
                assert event["source"].startswith("verdict")
        finally:
            httpd.shutdown()

    def test_trajectory_limit(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(10):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a/trajectory?limit=3")
            assert status == 200
            assert len(data["events"]) <= 3
        finally:
            httpd.shutdown()

    def test_trust_report_with_fingerprint(self):
        httpd, runtime, port = _setup_server()
        try:
            for _ in range(10):
                _evaluate(runtime, "a", "read", "/data")

            status, data = _get(port, "/v1/trust/a")
            assert status == 200
            assert "fingerprint" in data
            assert data["fingerprint"]["total_observations"] == 10
        finally:
            httpd.shutdown()
