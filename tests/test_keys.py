"""Tests for the key management module."""

from nomotic.keys import SigningKey, VerifyKey


class TestKeyGeneration:
    def test_generate_returns_key_pair(self):
        sk, vk = SigningKey.generate()
        assert isinstance(sk, SigningKey)
        assert isinstance(vk, VerifyKey)

    def test_generated_keys_are_32_bytes(self):
        sk, vk = SigningKey.generate()
        assert len(sk.to_bytes()) == 32
        assert len(vk.to_bytes()) == 32

    def test_each_generation_is_unique(self):
        sk1, vk1 = SigningKey.generate()
        sk2, vk2 = SigningKey.generate()
        assert sk1.to_bytes() != sk2.to_bytes()
        assert vk1.to_bytes() != vk2.to_bytes()


class TestSigningAndVerification:
    def test_sign_verify_round_trip(self):
        sk, vk = SigningKey.generate()
        data = b"hello governance"
        sig = sk.sign(data)
        assert vk.verify(sig, data)

    def test_wrong_data_fails_verification(self):
        sk, vk = SigningKey.generate()
        sig = sk.sign(b"original")
        assert not vk.verify(sig, b"tampered")

    def test_wrong_key_fails_verification(self):
        sk1, _vk1 = SigningKey.generate()
        _sk2, vk2 = SigningKey.generate()
        sig = sk1.sign(b"data")
        assert not vk2.verify(sig, b"data")

    def test_signature_is_deterministic(self):
        sk, _vk = SigningKey.generate()
        data = b"deterministic"
        sig1 = sk.sign(data)
        sig2 = sk.sign(data)
        assert sig1 == sig2

    def test_signature_is_32_bytes(self):
        sk, _vk = SigningKey.generate()
        sig = sk.sign(b"data")
        assert len(sig) == 32

    def test_empty_data(self):
        sk, vk = SigningKey.generate()
        sig = sk.sign(b"")
        assert vk.verify(sig, b"")


class TestKeySerialization:
    def test_signing_key_round_trip(self):
        sk, _vk = SigningKey.generate()
        raw = sk.to_bytes()
        restored = SigningKey.from_bytes(raw)
        assert sk.to_bytes() == restored.to_bytes()

    def test_verify_key_round_trip(self):
        _sk, vk = SigningKey.generate()
        raw = vk.to_bytes()
        restored = VerifyKey.from_bytes(raw)
        assert vk.to_bytes() == restored.to_bytes()

    def test_restored_key_can_verify(self):
        sk, vk = SigningKey.generate()
        sig = sk.sign(b"test")
        restored_vk = VerifyKey.from_bytes(vk.to_bytes())
        assert restored_vk.verify(sig, b"test")

    def test_restored_signing_key_produces_same_signature(self):
        sk, vk = SigningKey.generate()
        sig1 = sk.sign(b"test")
        restored = SigningKey.from_bytes(sk.to_bytes())
        sig2 = restored.sign(b"test")
        assert sig1 == sig2

    def test_invalid_key_length_raises(self):
        import pytest
        with pytest.raises(ValueError):
            SigningKey.from_bytes(b"too short")
        with pytest.raises(ValueError):
            VerifyKey.from_bytes(b"too short")


class TestFingerprint:
    def test_fingerprint_format(self):
        _sk, vk = SigningKey.generate()
        fp = vk.fingerprint()
        assert fp.startswith("SHA256:")
        hex_part = fp[len("SHA256:"):]
        assert len(hex_part) == 64  # SHA-256 hex is 64 chars

    def test_fingerprint_deterministic(self):
        _sk, vk = SigningKey.generate()
        assert vk.fingerprint() == vk.fingerprint()

    def test_different_keys_different_fingerprints(self):
        _sk1, vk1 = SigningKey.generate()
        _sk2, vk2 = SigningKey.generate()
        assert vk1.fingerprint() != vk2.fingerprint()

    def test_verify_key_derives_consistently(self):
        sk, _vk = SigningKey.generate()
        vk1 = sk.verify_key()
        vk2 = sk.verify_key()
        assert vk1.fingerprint() == vk2.fingerprint()
