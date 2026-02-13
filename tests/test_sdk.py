"""Tests for the Nomotic Agent SDK."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey, VerifyKey
from nomotic.sdk import (
    CertificateLoadError,
    GovernedAgent,
    GovernedRequestError,
    GovernedResponse,
)
from nomotic.store import MemoryCertificateStore


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_ca():
    """Create a CA and issue a certificate for testing."""
    issuer_sk, _ = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=issuer_sk, store=store)
    cert, agent_sk = ca.issue("test-agent", "customer-experience", "test-org", "global/us")
    return ca, cert, agent_sk


class EchoHandler(BaseHTTPRequestHandler):
    """Test server that echoes headers and body back as JSON."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        headers = {k: v for k, v in self.headers.items()}
        body = json.dumps({"method": "GET", "path": self.path, "headers": headers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items()}
        body = json.dumps({
            "method": "POST",
            "path": self.path,
            "headers": headers,
            "body": request_body.decode("utf-8", errors="replace"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items()}
        body = json.dumps({
            "method": "PUT",
            "path": self.path,
            "headers": headers,
            "body": request_body.decode("utf-8", errors="replace"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PATCH(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items()}
        body = json.dumps({
            "method": "PATCH",
            "path": self.path,
            "headers": headers,
            "body": request_body.decode("utf-8", errors="replace"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self):
        headers = {k: v for k, v in self.headers.items()}
        body = json.dumps({"method": "DELETE", "path": self.path, "headers": headers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ErrorHandler(BaseHTTPRequestHandler):
    """Test server that returns 4xx/5xx errors."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = b'{"error": "not_found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def echo_server():
    """Start a test echo server for the module."""
    server = HTTPServer(("127.0.0.1", 0), EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def error_server():
    """Start a test error server for the module."""
    server = HTTPServer(("127.0.0.1", 0), ErrorHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ── Loading tests ─────────────────────────────────────────────────────


class TestGovernedAgentLoading:
    def test_from_objects(self):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        assert agent.certificate is cert
        assert agent.cert_id == cert.certificate_id
        assert agent.trust_score == cert.trust_score
        assert agent.behavioral_age == cert.behavioral_age

    def test_from_files(self, tmp_path):
        _ca, cert, agent_sk = _make_ca()
        cert_path = tmp_path / f"{cert.certificate_id}.json"
        key_path = tmp_path / f"{cert.certificate_id}.key"
        cert_path.write_text(cert.to_json(), encoding="utf-8")
        key_path.write_bytes(agent_sk.to_bytes())

        agent = GovernedAgent.from_files(cert_path, key_path)
        assert agent.cert_id == cert.certificate_id

    def test_from_cert_id(self, tmp_path):
        _ca, cert, agent_sk = _make_ca()
        certs_dir = tmp_path / "certs"
        certs_dir.mkdir()
        (certs_dir / f"{cert.certificate_id}.json").write_text(cert.to_json())
        (certs_dir / f"{cert.certificate_id}.key").write_bytes(agent_sk.to_bytes())

        agent = GovernedAgent.from_cert_id(cert.certificate_id, base_dir=tmp_path)
        assert agent.cert_id == cert.certificate_id

    def test_from_files_missing_cert(self, tmp_path):
        with pytest.raises(CertificateLoadError, match="Failed to load certificate"):
            GovernedAgent.from_files(
                tmp_path / "nonexistent.json",
                tmp_path / "nonexistent.key",
            )

    def test_from_files_missing_key(self, tmp_path):
        _ca, cert, _agent_sk = _make_ca()
        cert_path = tmp_path / "test.json"
        cert_path.write_text(cert.to_json())
        with pytest.raises(CertificateLoadError, match="Failed to load signing key"):
            GovernedAgent.from_files(cert_path, tmp_path / "nonexistent.key")


# ── HTTP request tests ────────────────────────────────────────────────


class TestGovernedAgentRequests:
    def test_get_includes_nomotic_headers(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.get(f"{echo_server}/test")

        assert resp.ok
        assert resp.status == 200
        data = resp.json()
        headers = data["headers"]
        assert "X-Nomotic-Cert-ID" in headers
        assert "X-Nomotic-Trust" in headers
        assert "X-Nomotic-Age" in headers
        assert "X-Nomotic-Archetype" in headers
        assert "X-Nomotic-Zone" in headers
        assert "X-Nomotic-Signature" in headers
        assert "X-Nomotic-Gov-Hash" in headers

    def test_get_header_values(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.get(f"{echo_server}/test")
        data = resp.json()
        headers = data["headers"]
        assert headers["X-Nomotic-Cert-ID"] == cert.certificate_id
        assert headers["X-Nomotic-Archetype"] == cert.archetype

    def test_post_with_json(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.post(f"{echo_server}/data", json={"key": "value"})
        assert resp.ok
        data = resp.json()
        assert data["method"] == "POST"
        assert data["headers"].get("Content-Type") == "application/json"
        # Verify the body was serialized
        assert '"key":"value"' in data["body"]

    def test_post_with_data(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.post(f"{echo_server}/data", data=b"raw bytes")
        assert resp.ok
        data = resp.json()
        assert data["body"] == "raw bytes"

    def test_put_request(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.put(f"{echo_server}/data", json={"update": True})
        assert resp.ok
        data = resp.json()
        assert data["method"] == "PUT"

    def test_patch_request(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.patch(f"{echo_server}/data", json={"patch": True})
        assert resp.ok
        data = resp.json()
        assert data["method"] == "PATCH"

    def test_delete_request(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.delete(f"{echo_server}/data")
        assert resp.ok
        data = resp.json()
        assert data["method"] == "DELETE"

    def test_signature_is_valid(self, echo_server):
        """Verify that the signature in headers is valid for the body."""
        import base64
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)

        # POST with known body
        resp = agent.post(f"{echo_server}/data", json={"test": 1})
        data = resp.json()
        sig_b64 = data["headers"]["X-Nomotic-Signature"]
        sig = base64.b64decode(sig_b64)

        # The body that was signed is the canonical JSON
        signed_body = json.dumps({"test": 1}, sort_keys=True, separators=(",", ":")).encode()
        vk = VerifyKey.from_bytes(cert.public_key)
        assert vk.verify(sig, signed_body)


# ── URL resolution tests ─────────────────────────────────────────────


class TestURLResolution:
    def test_base_url_prepended(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(
            certificate=cert, signing_key=agent_sk, base_url=echo_server,
        )
        resp = agent.get("/test/path")
        assert resp.ok
        data = resp.json()
        assert data["path"] == "/test/path"

    def test_absolute_url_ignores_base(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(
            certificate=cert, signing_key=agent_sk, base_url="http://should-not-use:9999",
        )
        resp = agent.get(f"{echo_server}/test")
        assert resp.ok

    def test_base_url_trailing_slash(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(
            certificate=cert, signing_key=agent_sk, base_url=echo_server + "/",
        )
        resp = agent.get("/test")
        assert resp.ok


# ── Extra headers ─────────────────────────────────────────────────────


class TestExtraHeaders:
    def test_extra_headers_included(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(
            certificate=cert, signing_key=agent_sk,
            extra_headers={"X-Custom": "hello"},
        )
        resp = agent.get(f"{echo_server}/test")
        data = resp.json()
        assert data["headers"].get("X-Custom") == "hello"


# ── GovernedResponse tests ────────────────────────────────────────────


class TestGovernedResponse:
    def test_ok_for_2xx(self):
        resp = GovernedResponse(status=200, headers={}, body=b"", url="http://test")
        assert resp.ok
        resp = GovernedResponse(status=201, headers={}, body=b"", url="http://test")
        assert resp.ok
        resp = GovernedResponse(status=299, headers={}, body=b"", url="http://test")
        assert resp.ok

    def test_not_ok_for_4xx(self):
        resp = GovernedResponse(status=404, headers={}, body=b"", url="http://test")
        assert not resp.ok

    def test_not_ok_for_5xx(self):
        resp = GovernedResponse(status=500, headers={}, body=b"", url="http://test")
        assert not resp.ok

    def test_json_parsing(self):
        resp = GovernedResponse(
            status=200, headers={}, body=b'{"key": "value"}', url="http://test",
        )
        assert resp.json() == {"key": "value"}

    def test_text_decoding(self):
        resp = GovernedResponse(
            status=200, headers={}, body="hello".encode("utf-8"), url="http://test",
        )
        assert resp.text == "hello"

    def test_http_error_returns_response(self, error_server):
        """HTTP 4xx/5xx are returned as GovernedResponse, not raised."""
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        resp = agent.get(f"{error_server}/not-found")
        assert resp.status == 404
        assert not resp.ok


# ── Error handling tests ──────────────────────────────────────────────


class TestErrorHandling:
    def test_connection_refused_raises(self):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk, timeout=1.0)
        with pytest.raises(GovernedRequestError, match="failed"):
            agent.get("http://127.0.0.1:1")  # port 1 should refuse


# ── Thread safety tests ──────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_requests(self, echo_server):
        _ca, cert, agent_sk = _make_ca()
        agent = GovernedAgent(certificate=cert, signing_key=agent_sk)
        errors = []
        results = []

        def make_request(i):
            try:
                resp = agent.get(f"{echo_server}/thread/{i}")
                results.append(resp.status)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(s == 200 for s in results)
