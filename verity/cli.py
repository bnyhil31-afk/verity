"""
verity.cli
==========
Command-line interface for Verity.

Commands:
  verity init [--profile PROFILE] [--force] [--dry-run]
      Initialize Verity by generating an Ed25519 keypair and signing
      principles.yaml.

  verity status
      Show Verity status: principles integrity, profile, Python version,
      optional extras, and graph backend.

  verity connect <source-id> [--type TYPE]
      Register a data source connector. (Stub — implemented in Phase 6.)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

# cli.py lives at verity/cli.py; principles.yaml is at the repo root
# (one level above the package directory)
_PACKAGE_DIR = Path(__file__).parent          # verity/
_REPO_ROOT = _PACKAGE_DIR.parent              # repo root

PRINCIPLES_PATH = _REPO_ROOT / "principles.yaml"

# Private-key filenames per profile (personal and developer only)
_KEY_FILENAME: dict[str, str] = {
    "personal": "verity_personal.pem",
    "developer": "verity_dev.pem",
}


# ── Helpers: key directory ────────────────────────────────────────────────────

def _key_dir() -> Path:
    """Return the Verity keys directory (~/.verity/keys/).

    Extracted into its own function so tests can monkeypatch it without
    touching the real home directory.
    """
    return Path.home() / ".verity" / "keys"


# ── Helpers: canonical content hash ──────────────────────────────────────────

def _canonical_hash(data: dict[str, Any]) -> str:
    """SHA-256 of the canonical principles content.

    Excludes ``signed_by`` and ``signature`` fields so the hash is identical
    whether the file is unsigned, signed, or re-signed.  Both ``cmd_init``
    (signing) and ``cmd_status`` (verification) use this same function, which
    guarantees they agree on what was signed.
    """
    signing_data = {k: v for k, v in data.items() if k not in ("signature", "signed_by")}
    canonical = json.dumps(signing_data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Helpers: Ed25519 key operations ──────────────────────────────────────────

def _generate_ed25519_keypair() -> tuple[Any, Any]:
    """Generate a fresh Ed25519 keypair.  Returns (private_key, public_key)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def _private_key_to_pem(private_key: Any) -> bytes:
    """Serialize a private key to unencrypted PKCS8 PEM bytes."""
    from cryptography.hazmat.primitives import serialization

    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_key_to_pem(public_key: Any) -> str:
    """Serialize a public key to SubjectPublicKeyInfo PEM (ASCII string)."""
    from cryptography.hazmat.primitives import serialization

    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def _sign(private_key: Any, message: str) -> str:
    """Sign *message* with *private_key*.  Returns a base64-encoded signature."""
    return base64.b64encode(private_key.sign(message.encode("utf-8"))).decode("ascii")


def _verify_sig(signed_by_pem: str, sig_b64: str, message: str) -> bool:
    """Verify an Ed25519 signature.  Returns True iff the signature is valid."""
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        pub_key = load_pem_public_key(signed_by_pem.encode("utf-8"))
        pub_key.verify(base64.b64decode(sig_b64), message.encode("utf-8"))
        return True
    except Exception:  # InvalidSignature or any parse error
        return False


# ── Helpers: YAML patching ────────────────────────────────────────────────────

def _load_principles() -> tuple[dict[str, Any], str]:
    """Load principles.yaml.  Returns ``(parsed_dict, raw_file_content)``."""
    content = PRINCIPLES_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(content), content  # type: ignore[return-value]


def _patch_yaml_field(content: str, key: str, yaml_value: str) -> str:
    """Replace a top-level YAML field value, preserving any inline comment.

    ``yaml_value`` must be the YAML scalar representation of the new value
    (e.g. a bare string like ``abcdef==`` or a double-quoted scalar like
    ``"-----BEGIN PUBLIC KEY-----\\\\n..."``).

    The heuristic for detecting an inline comment is: two or more whitespace
    characters immediately followed by ``#``.  This is robust enough for the
    specific lines in principles.yaml and does not require a full YAML parser.
    """
    result: list[str] = []
    for line in content.splitlines(keepends=True):
        if line.startswith(f"{key}:"):
            # Detect inline comment: 2+ whitespace chars then '#'
            m = re.search(r"\s{2,}#", line)
            comment = line[m.start() :].rstrip("\n") if m else ""
            result.append(f"{key}: {yaml_value}{comment}\n")
        else:
            result.append(line)
    return "".join(result)


def _pem_to_yaml_scalar(pem: str) -> str:
    """Convert a PEM string to a YAML double-quoted scalar on a single line.

    In YAML double-quoted scalars ``\\n`` is an escape for a real newline, so
    ``yaml.safe_load`` will reconstruct the multiline PEM correctly.
    """
    escaped = pem.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


# ── Helpers: optional-package detection ──────────────────────────────────────

def _is_installed(package: str) -> bool:
    """Return True if *package* can be imported (i.e. is installed)."""
    import importlib

    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


# ── Command implementations ───────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> int:
    """``verity init`` — generate a keypair and sign principles.yaml."""
    profile: str = args.profile
    dry_run: bool = args.dry_run
    force: bool = args.force

    # Professional / enterprise require a formal ceremony
    if profile in ("professional", "enterprise"):
        print(
            "Professional/enterprise initialization requires a key ceremony. "
            "See CONTRIBUTING.md for instructions."
        )
        return 0

    # Load current principles
    try:
        data, content = _load_principles()
    except FileNotFoundError:
        print("Error: principles.yaml not found. Is this a Verity repository?")
        return 1

    already_signed = data.get("signature") is not None

    if already_signed and not force:
        print("✓ Already initialized. Use --force to re-initialize.")
        return 0

    # Determine where the private key will live
    key_path = _key_dir() / _KEY_FILENAME[profile]

    if dry_run:
        print("dry-run: would perform the following actions:")
        print("  Generate Ed25519 keypair")
        print(f"  Store private key at: {key_path}")
        print("  Set permissions: 600 (owner read/write only)")
        print("  Compute canonical hash of principles.yaml content")
        print("  Sign hash with Ed25519 private key")
        print("  Write public key to principles.yaml signed_by field")
        print("  Write base64 signature to principles.yaml signature field")
        return 0

    # Generate keypair
    private_key, public_key = _generate_ed25519_keypair()
    priv_pem = _private_key_to_pem(private_key)
    pub_pem = _public_key_to_pem(public_key)

    # Store private key with restricted permissions
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(priv_pem)
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    # Compute canonical hash over unsigned content (signed_by and signature
    # are explicitly nulled so the hash is stable before/after writing)
    signing_data = dict(data)
    signing_data["signature"] = None
    signing_data["signed_by"] = None
    content_hash = _canonical_hash(signing_data)
    sig_b64 = _sign(private_key, content_hash)

    # Patch principles.yaml in-place, preserving comments
    new_content = _patch_yaml_field(content, "signed_by", _pem_to_yaml_scalar(pub_pem))
    new_content = _patch_yaml_field(new_content, "signature", sig_b64)
    PRINCIPLES_PATH.write_text(new_content, encoding="utf-8")

    print(f"✓ Verity initialized ({profile} profile)")
    print(f"  Key: {key_path}")
    print("  Back up this key. Loss means re-initialization.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``verity status`` — print engine health and principles status."""
    # Load principles
    try:
        data, _ = _load_principles()
    except FileNotFoundError:
        print("Error: principles.yaml not found. Run: verity init")
        return 1

    signature = data.get("signature")
    signed_by = data.get("signed_by")

    # Determine signature status using the same canonical hash used at signing
    if signature is None or signed_by is None:
        sig_status = "unsigned"
        exit_code = 0
    else:
        signing_data = dict(data)
        signing_data["signature"] = None
        signing_data["signed_by"] = None
        expected_hash = _canonical_hash(signing_data)

        if _verify_sig(str(signed_by), str(signature), expected_hash):
            sig_status = "signed"
            exit_code = 0
        else:
            sig_status = "tampered"
            exit_code = 1

    # Determine graph backend (mirrors registry.py logic, without importing it)
    import os

    backend_env = os.getenv("VERITY_GRAPH_BACKEND", "rdflib").lower().strip()
    if backend_env not in ("rdflib", "pgvector", "jena"):
        backend_env = "rdflib"

    if backend_env == "rdflib":
        if _is_installed("oxrdflib"):
            graph_backend = "rdflib + Oxigraph (oxrdflib detected)"
        else:
            graph_backend = "rdflib (default)"
    else:
        graph_backend = backend_env

    print("Verity status")
    print(f"  principles:    {sig_status}")
    print("  profile:       personal (default)")
    print(f"  python:        {sys.version.split()[0]}")
    print(f"  pyoxigraph:    {'installed' if _is_installed('pyoxigraph') else 'not installed'}")
    print(f"  oxrdflib:      {'installed' if _is_installed('oxrdflib') else 'not installed'}")
    print(f"  graph backend: {graph_backend}")

    if sig_status == "tampered":
        print()
        print("  WARNING: signature does not match content.")
        print("           principles.yaml may have been modified after signing.")

    return exit_code


def cmd_connect(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``verity connect`` — register a data source connector (stub)."""
    print("Connector registration coming in Phase 6.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="verity",
        description="Verity — universal traversal and comprehension engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize Verity")
    init_parser.add_argument(
        "--profile",
        default="personal",
        choices=["personal", "developer", "professional", "enterprise"],
        help="Deployment profile (default: personal)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-initialize even if principles.yaml is already signed",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing any files",
    )

    # status
    subparsers.add_parser("status", help="Show Verity status")

    # connect
    connect_parser = subparsers.add_parser("connect", help="Connect a data source")
    connect_parser.add_argument("source_id", nargs="?", help="Source identifier")
    connect_parser.add_argument("--type", default=None, help="Connector type")

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "connect":
        sys.exit(cmd_connect(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
