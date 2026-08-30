from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xiangqi_agent.engine.assets import (
    AssetIntegrityError,
    AssetLockError,
    PikafishAssetLock,
    choose_compatible_executable,
    validate_archive_members,
    verify_asset,
)


def _write_lock(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "project": "official-pikafish/Pikafish",
        "tag": "Pikafish-2026-01-02",
        "commit": "ce0679e00ee196f7ba17f6ec18941b9a5036f8cf",
        "asset_name": "Pikafish.2026-01-02.7z",
        "url": (
            "https://github.com/official-pikafish/Pikafish/releases/download/"
            "Pikafish-2026-01-02/Pikafish.2026-01-02.7z"
        ),
        "size": 55_332_846,
        "sha256": "84257063905615919fb4ee6a70273a94843bb6ec04c45e3ac706098838bc1a49",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_asset_lock_accepts_only_pinned_official_release(tmp_path: Path) -> None:
    lock = PikafishAssetLock.load(_write_lock(tmp_path / "asset.json"))

    assert lock.tag == "Pikafish-2026-01-02"
    assert lock.commit.startswith("ce0679e")
    assert lock.asset_name == "Pikafish.2026-01-02.7z"


def test_repository_asset_lock_pins_the_reviewed_release() -> None:
    lock_path = Path(__file__).parents[3] / "assets" / "pikafish-2026-01-02.json"

    lock = PikafishAssetLock.load(lock_path)

    assert lock.tag == "Pikafish-2026-01-02"
    assert lock.commit == "ce0679e00ee196f7ba17f6ec18941b9a5036f8cf"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://example.com/Pikafish.7z"),
        ("url", "http://github.com/official-pikafish/Pikafish/releases/download/x/y"),
        ("project", "someone/fork"),
        ("sha256", "not-a-digest"),
        ("commit", "short"),
        ("size", 0),
    ],
)
def test_asset_lock_rejects_untrusted_or_unpinned_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(AssetLockError):
        PikafishAssetLock.load(_write_lock(tmp_path / "asset.json", **{field: value}))


def test_verify_asset_checks_both_size_and_sha256(tmp_path: Path) -> None:
    archive = tmp_path / "engine.7z"
    archive.write_bytes(b"official archive bytes")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    lock = PikafishAssetLock.load(
        _write_lock(tmp_path / "asset.json", size=archive.stat().st_size, sha256=digest)
    )

    verify_asset(archive, lock)
    archive.write_bytes(b"tampered archive bytes")

    with pytest.raises(AssetIntegrityError):
        verify_asset(archive, lock)


@pytest.mark.parametrize(
    "member",
    ["../escape.exe", "folder/../../escape.exe", "/absolute.exe", r"C:\escape.exe"],
)
def test_archive_member_validation_rejects_path_escape(member: str) -> None:
    with pytest.raises(AssetIntegrityError):
        validate_archive_members(["Pikafish/good.exe", member])


def test_archive_member_validation_accepts_relative_tree() -> None:
    validate_archive_members(
        [
            "Pikafish/",
            "Pikafish/Windows/",
            "Pikafish/Windows/pikafish-avx2.exe",
            "Pikafish/pikafish.nnue",
        ]
    )


def test_choose_compatible_executable_uses_a_real_probe_and_falls_back(tmp_path: Path) -> None:
    first = tmp_path / "pikafish-avx512.exe"
    second = tmp_path / "pikafish-avx2.exe"
    first.touch()
    second.touch()
    attempted: list[str] = []

    def probe(path: Path) -> bool:
        attempted.append(path.name)
        return path == second

    selected = choose_compatible_executable([first, second], probe)

    assert selected == second
    assert attempted == ["pikafish-avx512.exe", "pikafish-avx2.exe"]


def test_choose_compatible_executable_fails_closed_when_none_start(tmp_path: Path) -> None:
    candidate = tmp_path / "pikafish.exe"
    candidate.touch()

    with pytest.raises(AssetIntegrityError, match="compatible"):
        choose_compatible_executable([candidate], lambda _: False)
