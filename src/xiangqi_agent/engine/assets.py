from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse


class AssetLockError(ValueError):
    """Pinned engine metadata is missing, malformed, or untrusted."""


class AssetIntegrityError(RuntimeError):
    """A downloaded or extracted engine asset failed a safety check."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_OFFICIAL_PROJECT = "official-pikafish/Pikafish"
_OFFICIAL_HOST = "github.com"


@dataclass(frozen=True, slots=True)
class PikafishAssetLock:
    schema_version: int
    project: str
    tag: str
    commit: str
    asset_name: str
    url: str
    size: int
    sha256: str

    @classmethod
    def load(cls, path: Path) -> PikafishAssetLock:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            lock = cls(
                schema_version=payload["schema_version"],
                project=payload["project"],
                tag=payload["tag"],
                commit=payload["commit"],
                asset_name=payload["asset_name"],
                url=payload["url"],
                size=payload["size"],
                sha256=payload["sha256"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AssetLockError("asset lock is missing required metadata") from exc
        lock._validate()
        return lock

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise AssetLockError("unsupported asset lock schema")
        if self.project != _OFFICIAL_PROJECT:
            raise AssetLockError("asset must come from the official Pikafish project")
        if not isinstance(self.tag, str) or not self.tag.startswith("Pikafish-"):
            raise AssetLockError("Pikafish release tag is invalid")
        if not isinstance(self.commit, str) or _COMMIT.fullmatch(self.commit) is None:
            raise AssetLockError("Pikafish commit must be a full SHA-1")
        if not isinstance(self.asset_name, str) or Path(self.asset_name).name != self.asset_name:
            raise AssetLockError("asset name must be a plain filename")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise AssetLockError("asset size must be positive")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise AssetLockError("asset SHA-256 is invalid")
        parsed = urlparse(self.url)
        expected_prefix = f"/{self.project}/releases/download/{self.tag}/"
        if (
            parsed.scheme != "https"
            or parsed.netloc != _OFFICIAL_HOST
            or not parsed.path.startswith(expected_prefix)
            or parsed.path != expected_prefix + self.asset_name
            or parsed.query
            or parsed.fragment
        ):
            raise AssetLockError("asset URL is not the pinned official GitHub release URL")


def verify_asset(path: Path, lock: PikafishAssetLock) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AssetIntegrityError("engine archive is unavailable") from exc
    if size != lock.size:
        raise AssetIntegrityError(f"engine archive size mismatch: expected {lock.size}, got {size}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AssetIntegrityError("engine archive could not be read") from exc
    if digest.hexdigest() != lock.sha256:
        raise AssetIntegrityError("engine archive SHA-256 mismatch")


def validate_archive_members(members: Iterable[str]) -> None:
    for raw_name in members:
        if not raw_name or "\x00" in raw_name:
            raise AssetIntegrityError("archive contains an invalid member name")
        normalized = raw_name.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(raw_name)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or any(part == ".." for part in posix.parts)
        ):
            raise AssetIntegrityError(f"archive member escapes the destination: {raw_name}")


def choose_compatible_executable(
    candidates: Iterable[Path], probe: Callable[[Path], bool]
) -> Path:
    attempted = False
    for candidate in candidates:
        if candidate.suffix.lower() != ".exe" or not candidate.is_file():
            continue
        attempted = True
        if probe(candidate):
            return candidate
    detail = "none were present" if not attempted else "all candidates failed the UCI probe"
    raise AssetIntegrityError(f"no compatible Pikafish executable found: {detail}")
