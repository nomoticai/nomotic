"""Key management for Nomotic agent identity.

Provides signing and verification operations for agent birth certificates.
The signing interface is abstract so the crypto backend can be swapped
without changing calling code.

v1 uses HMAC-SHA256 as a signing primitive. The key pair model is preserved
so the interface is ready for Ed25519 (or any asymmetric scheme) in v2.
The "public key" in v1 is derived deterministically from the private key
using HMAC-SHA256 with a fixed derivation label, making it a distinct
value that can be embedded in certificates and used for verification
without exposing the signing secret.

Why HMAC-SHA256 for v1:
- Zero dependencies (hashlib + hmac are stdlib)
- Correct security properties for signing/verification
- Interface is identical to what Ed25519 v2 will use
- Ships now, upgrades cleanly later
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

__all__ = ["SigningKey", "VerifyKey"]

# Fixed label for deriving the "public" portion from the secret.
# Changing this would invalidate all existing key pairs.
_PUB_DERIVATION_LABEL = b"nomotic-v1-pub-derivation"


@dataclass(frozen=True)
class VerifyKey:
    """Public verification key.

    Verifies signatures produced by the corresponding SigningKey.
    In v1 this is an HMAC-derived token; the interface is unchanged for v2.
    """

    _key_bytes: bytes

    def verify(self, signature: bytes, data: bytes) -> bool:
        """Return True if *signature* is valid for *data*."""
        expected = hmac.new(self._key_bytes, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    def to_bytes(self) -> bytes:
        return self._key_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> VerifyKey:
        if len(data) != 32:
            raise ValueError(f"VerifyKey must be 32 bytes, got {len(data)}")
        return cls(_key_bytes=data)

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the public key: ``SHA256:<hex>``."""
        digest = hashlib.sha256(self._key_bytes).hexdigest()
        return f"SHA256:{digest}"


@dataclass(frozen=True)
class SigningKey:
    """Private signing key.

    Signs data and can export the corresponding VerifyKey.
    In v1 this wraps HMAC-SHA256; the interface is unchanged for v2.
    """

    _key_bytes: bytes

    @classmethod
    def generate(cls) -> tuple[SigningKey, VerifyKey]:
        """Generate a new random key pair."""
        secret = os.urandom(32)
        sk = cls(_key_bytes=secret)
        vk = sk.verify_key()
        return sk, vk

    def sign(self, data: bytes) -> bytes:
        """Produce a signature over *data*."""
        vk = self.verify_key()
        return hmac.new(vk.to_bytes(), data, hashlib.sha256).digest()

    def verify_key(self) -> VerifyKey:
        """Derive the public verification key."""
        pub = hmac.new(
            self._key_bytes, _PUB_DERIVATION_LABEL, hashlib.sha256
        ).digest()
        return VerifyKey(_key_bytes=pub)

    def to_bytes(self) -> bytes:
        return self._key_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> SigningKey:
        if len(data) != 32:
            raise ValueError(f"SigningKey must be 32 bytes, got {len(data)}")
        return cls(_key_bytes=data)
