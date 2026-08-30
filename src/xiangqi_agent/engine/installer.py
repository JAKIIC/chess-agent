from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory

import httpx

from xiangqi_agent.engine.assets import (
    AssetIntegrityError,
    PikafishAssetLock,
    choose_compatible_executable,
    validate_archive_members,
    verify_asset,
)
from xiangqi_agent.engine.process import EngineProcessError, PikafishProcess

ChunkSource = Callable[[str], Iterable[bytes]]
CommandRunner = Callable[[list[str]], CompletedProcess[str]]
EngineProbe = Callable[[Path, Path], bool]

_WINDOWS_BUILD_ORDER = (
    "pikafish-avx512icl.exe",
    "pikafish-vnni512.exe",
    "pikafish-avxvnni.exe",
    "pikafish-avx512.exe",
    "pikafish-bmi2.exe",
    "pikafish-avx2.exe",
    "pikafish-sse41-popcnt.exe",
)


@dataclass(frozen=True, slots=True)
class InstalledPikafish:
    tag: str
    commit: str
    executable: Path
    eval_file: Path
    asset_sha256: str


def http_chunks(url: str) -> Iterator[bytes]:
    with (
        httpx.Client(follow_redirects=True, timeout=60.0) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        yield from response.iter_bytes(chunk_size=1024 * 1024)


def download_verified_asset(
    lock: PikafishAssetLock,
    destination: Path,
    chunks: ChunkSource = http_chunks,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / lock.asset_name
    if final_path.is_file():
        verify_asset(final_path, lock)
        return final_path
    partial_path = destination / f"{lock.asset_name}.partial"
    partial_path.unlink(missing_ok=True)
    try:
        with partial_path.open("xb") as stream:
            for chunk in chunks(lock.url):
                stream.write(chunk)
        verify_asset(partial_path, lock)
        partial_path.replace(final_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return final_path


def extract_verified_archive(
    archive: Path,
    destination: Path,
    *,
    runner: CommandRunner | None = None,
    archive_tool: str = "tar",
) -> Path:
    if destination.exists():
        raise AssetIntegrityError(f"engine install destination already exists: {destination}")
    if not archive.is_file():
        raise AssetIntegrityError("engine archive is unavailable for extraction")
    run = _run_command if runner is None else runner
    listing = run([archive_tool, "-tf", str(archive)])
    _require_command_success(listing, "list engine archive")
    members = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not members:
        raise AssetIntegrityError("engine archive is empty")
    validate_archive_members(members)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pikafish-extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        extraction = run([archive_tool, "-xf", str(archive), "-C", str(stage)])
        _require_command_success(extraction, "extract engine archive")
        if not any(stage.iterdir()):
            raise AssetIntegrityError("engine archive extraction produced no files")
        stage.replace(destination)
    return destination


def find_windows_executables(install_root: Path) -> tuple[Path, ...]:
    windows_root = install_root / "Windows"
    found = {path.name.lower(): path for path in windows_root.glob("pikafish-*.exe")}
    ordered = [found[name] for name in _WINDOWS_BUILD_ORDER if name in found]
    ordered.extend(path for name, path in sorted(found.items()) if name not in _WINDOWS_BUILD_ORDER)
    return tuple(ordered)


def probe_pikafish(executable: Path, eval_file: Path) -> bool:
    engine = PikafishProcess(
        executable,
        threads=1,
        hash_mb=64,
        eval_file=eval_file,
        startup_timeout=8.0,
        shutdown_timeout=2.0,
    )
    try:
        engine.start()
    except (OSError, EngineProcessError):
        return False
    finally:
        engine.close()
    return True


def install_pikafish(
    lock_path: Path,
    local_root: Path,
    *,
    chunks: ChunkSource = http_chunks,
    runner: CommandRunner | None = None,
    probe: EngineProbe = probe_pikafish,
) -> InstalledPikafish:
    lock = PikafishAssetLock.load(lock_path)
    archive = download_verified_asset(lock, local_root / "downloads", chunks)
    install_root = local_root / lock.tag
    if not install_root.exists():
        extract_verified_archive(archive, install_root, runner=runner)
    required = (
        install_root / "Copying.txt",
        install_root / "NNUE-License.md",
        install_root / "pikafish.nnue",
    )
    if any(not path.is_file() for path in required):
        raise AssetIntegrityError("extracted engine is missing its license or NNUE bundle")
    eval_file = required[-1]
    executable = choose_compatible_executable(
        find_windows_executables(install_root), lambda candidate: probe(candidate, eval_file)
    )
    installed = InstalledPikafish(
        tag=lock.tag,
        commit=lock.commit,
        executable=executable.resolve(),
        eval_file=eval_file.resolve(),
        asset_sha256=lock.sha256,
    )
    _write_install_manifest(local_root, installed)
    return installed


def load_installed_pikafish(local_root: Path) -> InstalledPikafish:
    manifest_path = local_root / "current.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1:
            raise AssetIntegrityError("unsupported Pikafish install manifest")
        executable = _resolve_local_path(local_root, payload["executable"])
        eval_file = _resolve_local_path(local_root, payload["eval_file"])
        installed = InstalledPikafish(
            tag=payload["tag"],
            commit=payload["commit"],
            executable=executable,
            eval_file=eval_file,
            asset_sha256=payload["asset_sha256"],
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AssetIntegrityError("Pikafish install manifest is invalid") from exc
    if not installed.executable.is_file() or not installed.eval_file.is_file():
        raise AssetIntegrityError("Pikafish install manifest points to missing files")
    return installed


def _write_install_manifest(local_root: Path, installed: InstalledPikafish) -> None:
    root = local_root.resolve()
    payload = {
        "schema_version": 1,
        "tag": installed.tag,
        "commit": installed.commit,
        "asset_sha256": installed.asset_sha256,
        "executable": installed.executable.relative_to(root).as_posix(),
        "eval_file": installed.eval_file.relative_to(root).as_posix(),
    }
    local_root.mkdir(parents=True, exist_ok=True)
    manifest = local_root / "current.json"
    partial = local_root / "current.json.partial"
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(manifest)


def _resolve_local_path(local_root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise AssetIntegrityError("Pikafish install path must be text")
    root = local_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AssetIntegrityError("Pikafish install path escapes the local root") from exc
    return candidate


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )


def _require_command_success(result: CompletedProcess[str], action: str) -> None:
    if result.returncode == 0:
        return
    detail = result.stderr.strip() or "unknown archive tool error"
    raise AssetIntegrityError(f"could not {action}: {detail}")
