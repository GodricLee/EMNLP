"""Utility for generating a simple XOR-based check code to guard against hallucinated outputs."""

from __future__ import annotations

import hashlib
from typing import Callable


def _hash_hex(value: str, algo: str = "sha256") -> str:
	"""Return lowercase hex digest of the given value using the chosen algo."""

	digest_func: Callable[[bytes], "hashlib._Hash"] = getattr(hashlib, algo)
	return digest_func(value.encode("utf-8")).hexdigest()


def generate_check_code(id_part: str, value: str, algo: str = "sha256") -> str:
	"""
	Generate a check code by XOR-ing a suffix of the value hash with the provided id.

	Parameters
	----------
	id_part: str
		A short identifier derived from the hash prefix of ``value`` (e.g., first N hex chars).
	value: str
		The original value to verify.
	algo: str
		Hash algorithm name accepted by ``hashlib`` (default: ``sha256``).

	Returns
	-------
	str
		Hex string of length ``len(id_part)`` representing the XOR of ``id_part`` and the hash
		suffix of ``value``.

	Notes
	-----
	This uses a symmetric XOR between the id (hash prefix) and the hash suffix of the same value,
	making it easy to recompute and verify that generated content aligns with its source.
	"""

	if not id_part:
		raise ValueError("id_part must be non-empty")

	full_hash = _hash_hex(value, algo)
	suffix_len = len(id_part)
	hash_suffix = full_hash[-suffix_len:]

	# Ensure both strings are valid hex and same length
	if len(hash_suffix) != suffix_len:
		raise ValueError("hash suffix length mismatch")

	try:
		id_int = int(id_part, 16)
		suffix_int = int(hash_suffix, 16)
	except ValueError as exc:
		raise ValueError("id_part and hash suffix must be hex strings") from exc

	xor_result = id_int ^ suffix_int
	return f"{xor_result:0{suffix_len}x}"  # zero-pad to preserve length


__all__ = ["generate_check_code"]

if __name__ == "__main__":
	import sys

	test_args = [
    ("1ca43c", "The quick brown fox jumps over the lazy dog", "sha256"),
    ("abcdef", "Hello, World!", "md5"),("","")]

	for args in test_args:
		print(generate_check_code(*args))