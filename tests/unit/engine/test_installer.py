from __future__ import annotations

import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from xiangqi_agent.engine.assets import AssetIntegrityError, PikafishAssetLock
from xiangqi_agent.engine.installer import (
    download_verified_asset,
    extract_verified_archive,
    find_windows_executables,
    install_pikafish,
    load_installed_pikafish,
)


class _Chunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __call__(self, url: str) -> list[bytes]:
        assert url.startswith("https://github.com/official-pikafish/")
        return self._chunks


def _lock(path: Path, body: bytes, *, expected: bytes | None = None) -> PikafishAssetLock:
    checked = body if expected is None else expected
    payload = {
        "schema_version": 1,
        "project": "official-pikafish/Pikafish",
        "tag": "Pikafish-test",
        "commit": "1" * 40,
        "asset_name": "Pikafish.test.7z",
        "url": (
            "https://github.com/official-pikafish/Pikafish/releases/download/"
            "Pikafish-test/Pikafish.test.7z"
        ),
        "size": len(checked),
        "sha256": hashlib.sha256(checked).hexdigest(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return PikafishAssetLock.load(path)


def test_download_verified_asset_streams_to_atomic_final_path(tmp_path: Path) -> None:
    body = b"pinned engine archive"
    lock = _lock(tmp_path / "lock.json", body)

    result = download_verified_asset(lock, tmp_path / "download", _Chunks([body[:5], body[5:]]))

    assert result.read_bytes() == body
    assert result.name == lock.asset_name
    assert not list((tmp_path / "download").glob("*.partial"))


def test_download_verified_asset_removes_partial_file_on_integrity_failure(tmp_path: Path) -> None:
    expected = b"expected"
    received = b"tampered"
    lock = _lock(tmp_path / "lock.json", received, expected=expected)

    with pytest.raises(AssetIntegrityError):
        download_verified_asset(lock, tmp_path / "download", _Chunks([received]))

    assert not list((tmp_path / "download").iterdir())


def test_extract_verified_archive_validates_listing_before_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "engine.7z"
    archive.touch()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="Windows/good.exe\n../escape.exe\n", stderr="")

    with pytest.raises(AssetIntegrityError, match="escapes"):
        extract_verified_archive(archive, tmp_path / "installed", runner=runner)

    assert len(calls) == 1
    assert not (tmp_path / "installed").exists()


def test_extract_verified_archive_uses_staging_then_publishes_atomically(tmp_path: Path) -> None:
    archive = tmp_path / "engine.7z"
    archive.touch()
    destination = tmp_path / "installed"

    def runner(command: list[str]) -> CompletedProcess[str]:
        if "-tf" in command:
            return CompletedProcess(
                command,
                0,
                stdout="Windows/pikafish-sse41-popcnt.exe\npikafish.nnue\n",
                stderr="",
            )
        extract_root = Path(command[command.index("-C") + 1])
        (extract_root / "Windows").mkdir(parents=True)
        (extract_root / "Windows" / "pikafish-sse41-popcnt.exe").touch()
        (extract_root / "pikafish.nnue").touch()
        return CompletedProcess(command, 0, stdout="", stderr="")

    result = extract_verified_archive(archive, destination, runner=runner)

    assert result == destination
    assert (destination / "Windows" / "pikafish-sse41-popcnt.exe").is_file()
    assert (destination / "pikafish.nnue").is_file()
    assert not list(tmp_path.glob("pikafish-extract-*"))


def test_extract_verified_archive_refuses_to_overwrite_existing_install(tmp_path: Path) -> None:
    archive = tmp_path / "engine.7z"
    archive.touch()
    destination = tmp_path / "installed"
    destination.mkdir()

    with pytest.raises(AssetIntegrityError, match="already exists"):
        extract_verified_archive(archive, destination)


def test_find_windows_executables_returns_fastest_first_and_ignores_other_platforms(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "Windows"
    linux = tmp_path / "Linux"
    windows.mkdir()
    linux.mkdir()
    for name in ("pikafish-sse41-popcnt.exe", "pikafish-avx2.exe", "pikafish-avx512icl.exe"):
        (windows / name).touch()
    (linux / "pikafish-avx512.exe").touch()

    candidates = find_windows_executables(tmp_path)

    assert [path.name for path in candidates] == [
        "pikafish-avx512icl.exe",
        "pikafish-avx2.exe",
        "pikafish-sse41-popcnt.exe",
    ]


def test_install_pikafish_probes_candidates_and_writes_portable_local_manifest(
    tmp_path: Path,
) -> None:
    body = b"verified archive"
    lock = _lock(tmp_path / "lock.json", body)
    local_root = tmp_path / "pikafish"
    downloads = local_root / "downloads"
    downloads.mkdir(parents=True)
    (downloads / lock.asset_name).write_bytes(body)
    install_root = local_root / lock.tag
    windows = install_root / "Windows"
    windows.mkdir(parents=True)
    incompatible = windows / "pikafish-avx512icl.exe"
    selected = windows / "pikafish-avx2.exe"
    incompatible.touch()
    selected.touch()
    (install_root / "pikafish.nnue").write_bytes(b"network")
    (install_root / "Copying.txt").write_text("GPLv3", encoding="utf-8")
    (install_root / "NNUE-License.md").write_text("legal use", encoding="utf-8")

    installed = install_pikafish(
        tmp_path / "lock.json",
        local_root,
        probe=lambda path, _: path == selected,
    )
    loaded = load_installed_pikafish(local_root)

    assert installed.executable == selected.resolve()
    assert loaded == installed
    manifest = json.loads((local_root / "current.json").read_text(encoding="utf-8"))
    assert manifest["executable"] == "Pikafish-test/Windows/pikafish-avx2.exe"
    assert manifest["eval_file"] == "Pikafish-test/pikafish.nnue"
    assert str(tmp_path) not in (local_root / "current.json").read_text(encoding="utf-8")


def test_install_pikafish_rejects_incomplete_extracted_license_bundle(tmp_path: Path) -> None:
    body = b"verified archive"
    lock = _lock(tmp_path / "lock.json", body)
    local_root = tmp_path / "pikafish"
    downloads = local_root / "downloads"
    downloads.mkdir(parents=True)
    (downloads / lock.asset_name).write_bytes(body)
    install_root = local_root / lock.tag
    (install_root / "Windows").mkdir(parents=True)
    (install_root / "Windows" / "pikafish-avx2.exe").touch()
    (install_root / "pikafish.nnue").write_bytes(b"network")

    with pytest.raises(AssetIntegrityError, match="license"):
        install_pikafish(tmp_path / "lock.json", local_root, probe=lambda *_: True)
