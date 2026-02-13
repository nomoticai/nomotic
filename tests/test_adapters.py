"""Tests for framework adapters (import error paths)."""

import pytest


class TestFastAPIAdapter:
    def test_import_without_fastapi_raises(self):
        """Importing the FastAPI adapter without fastapi should raise ImportError."""
        # This test only runs if fastapi is NOT installed
        try:
            import fastapi  # noqa: F401
            pytest.skip("fastapi is installed, cannot test import error")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="pip install"):
            import nomotic.adapters.fastapi_adapter  # noqa: F401


class TestFlaskAdapter:
    def test_import_without_flask_raises(self):
        """Importing the Flask adapter without flask should raise ImportError."""
        try:
            import flask  # noqa: F401
            pytest.skip("flask is installed, cannot test import error")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="pip install"):
            import nomotic.adapters.flask_adapter  # noqa: F401
