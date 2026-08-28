from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock
from time import monotonic, sleep

import cv2

from xiangqi_agent.capture.probe import (
    CaptureProbeError,
    analyze_change_sequence,
    summarize_capture,
)
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.capture.windows_capture_source import WindowsCaptureSource
from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.platform.windows import (
    WindowInfo,
    WindowSelectionError,
    WindowsWindowCatalog,
    select_window,
)
from xiangqi_agent.vision.geometry import BoardGeometry, GeometryError, parse_normalized_quad


def main() -> int:
    args = _parse_args()
    try:
        candidates = WindowsWindowCatalog().list_candidates()
        if args.list_windows:
            _print_candidates(candidates)
            return 0 if candidates else 2
        window = _choose_window(candidates, args.hwnd)
        frames, close_errors = _capture(window, args.seconds, args.fps)
        capture = summarize_capture(frames)
        result: dict[str, object] = {
            "capture": {
                "frame_count": capture.frame_count,
                "first_size": capture.first_size,
                "last_size": capture.last_size,
                "size_change_count": capture.size_change_count,
                "effective_fps": round(capture.effective_fps, 3),
                "timestamps_monotonic": capture.timestamps_monotonic,
                "duration_seconds": round(capture.duration_seconds, 3),
                "capture_closed": bool(close_errors),
            }
        }
        quad_text = args.quad
        if args.prompt_quad:
            debug_dir = _require_debug_dir(args.debug_dir)
            preview = debug_dir / "first-frame.png"
            _write_image(preview, frames[0].bgra)
            print(f"Preview saved to ignored debug path: {preview}")
            quad_text = input("Enter normalized TL;TR;BR;BL as x,y;x,y;x,y;x,y: ").strip()
        if quad_text:
            geometry = BoardGeometry.from_quad(
                parse_normalized_quad(quad_text),
                frames[0].size,
                orientation=Orientation(args.orientation),
            )
            points = geometry.grid_points()
            patches = geometry.crop_intersections(frames[0].bgra)
            changes = analyze_change_sequence(
                frames,
                geometry,
                global_threshold=args.global_threshold,
                local_threshold=args.local_threshold,
                top_k=args.top_k,
            )
            result["geometry"] = {
                "point_count": len(points),
                "first_point": tuple(round(value, 2) for value in points[0]),
                "last_point": tuple(round(value, 2) for value in points[-1]),
                "patch_count": len(patches),
                "patch_shape": patches[0].shape,
            }
            result["change"] = {
                "comparison_count": changes.comparison_count,
                "stable_comparison_count": changes.stable_comparison_count,
                "trailing_stable_comparisons": changes.trailing_stable_comparisons,
                "peak_global_difference": round(changes.peak_global_difference, 4),
                "most_changed": tuple(
                    (index, round(score, 4)) for index, score in changes.most_changed
                ),
            }
            if args.debug_dir is not None:
                debug_dir = _require_debug_dir(args.debug_dir)
                overlay = frames[0].bgra.copy()
                overlay.setflags(write=True)
                for index, (x, y) in enumerate(points):
                    color = (0, 255, 0, 255) if index not in (0, 89) else (0, 0, 255, 255)
                    cv2.circle(overlay, (round(x), round(y)), 4, color, thickness=1)
                _write_image(debug_dir / "calibrated-grid.png", overlay)
        else:
            result["geometry"] = {"status": "skipped; provide --quad or --prompt-quad"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CaptureClosedError, CaptureProbeError, GeometryError, WindowSelectionError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-second visible-window capture and board-geometry probe")
    parser.add_argument("--list-windows", action="store_true")
    parser.add_argument("--hwnd", type=int)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--quad", help="normalized TL;TR;BR;BL coordinates")
    parser.add_argument("--prompt-quad", action="store_true")
    parser.add_argument("--orientation", choices=tuple(item.value for item in Orientation), default="red_bottom")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--global-threshold", type=float, default=1.5)
    parser.add_argument("--local-threshold", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 10:
        parser.error("--seconds must be greater than zero and at most ten")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.prompt_quad and args.quad:
        parser.error("use only one of --prompt-quad and --quad")
    if args.prompt_quad and args.debug_dir is None:
        parser.error("--prompt-quad requires --debug-dir")
    return args


def _choose_window(candidates: tuple[WindowInfo, ...], hwnd: int | None) -> WindowInfo:
    if hwnd is not None:
        return select_window(candidates, hwnd)
    if not candidates:
        raise WindowSelectionError("no candidate target windows are available")
    _print_candidates(candidates)
    try:
        index = int(input("Select candidate number: ").strip())
        return candidates[index]
    except (ValueError, IndexError) as exc:
        raise WindowSelectionError("manual window selection is invalid") from exc


def _print_candidates(candidates: tuple[WindowInfo, ...]) -> None:
    for index, item in enumerate(candidates):
        print(
            f"[{index}] hwnd={item.hwnd} process={item.process_name} "
            f"size={item.client_size[0]}x{item.client_size[1]} title={item.title}"
        )


def _capture(
    window: WindowInfo,
    seconds: float,
    fps: int,
) -> tuple[tuple[CaptureFrame, ...], tuple[CaptureClosedError, ...]]:
    frames: list[CaptureFrame] = []
    close_errors: list[CaptureClosedError] = []
    lock = Lock()

    def receive(frame: CaptureFrame) -> None:
        with lock:
            frames.append(frame)

    def closed(error: CaptureClosedError) -> None:
        with lock:
            close_errors.append(error)

    source = WindowsCaptureSource(window, fps=fps)
    source.start(receive, closed)
    deadline = monotonic() + seconds
    try:
        while monotonic() < deadline:
            with lock:
                if close_errors:
                    break
            sleep(min(0.05, max(0.0, deadline - monotonic())))
    finally:
        source.close()
    with lock:
        return tuple(frames), tuple(close_errors)


def _require_debug_dir(path: Path | None) -> Path:
    if path is None:
        raise GeometryError("an ignored debug directory is required")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_image(path: Path, pixels: object) -> None:
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"failed to write debug image {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
