from __future__ import annotations

import numpy as np

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.monitor import CaptureMonitor, MonitorStatus
from xiangqi_agent.vision.geometry import parse_normalized_quad

QUAD = parse_normalized_quad("0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9")


def test_monitor_reports_first_valid_frame_and_ninety_points() -> None:
    source = FakeFrameSource(hwnd=42)
    updates = []
    monitor = CaptureMonitor(source, QUAD, on_update=updates.append)

    monitor.start()
    source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)
    monitor.close()
    monitor.close()

    assert [update.status for update in updates] == [
        MonitorStatus.CONNECTING,
        MonitorStatus.WATCHING,
    ]
    assert updates[-1].frame_size == (300, 200)
    assert updates[-1].point_count == 90


def test_monitor_rebinds_calibration_on_proportional_frame_size_change() -> None:
    source = FakeFrameSource(hwnd=42)
    updates = []
    monitor = CaptureMonitor(source, QUAD, on_update=updates.append)
    monitor.start()
    source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)

    source.push(np.zeros((300, 450, 4), dtype=np.uint8), timestamp_ns=2)

    assert updates[-1].status is MonitorStatus.GEOMETRY_REBOUND
    assert updates[-1].frame_size == (450, 300)
    assert updates[-1].point_count == 90
    monitor.close()


def test_monitor_invalidates_calibration_on_material_aspect_ratio_change() -> None:
    source = FakeFrameSource(hwnd=42)
    updates = []
    monitor = CaptureMonitor(source, QUAD, on_update=updates.append)
    monitor.start()
    source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)

    source.push(np.zeros((200, 450, 4), dtype=np.uint8), timestamp_ns=2)

    assert updates[-1].status is MonitorStatus.GEOMETRY_INVALID
    assert "aspect" in updates[-1].message.lower()
    monitor.close()


def test_monitor_reports_target_close_and_ignores_duplicate_close() -> None:
    source = FakeFrameSource(hwnd=42)
    updates = []
    monitor = CaptureMonitor(source, QUAD, on_update=updates.append)
    monitor.start()

    source.simulate_target_close()
    source.simulate_target_close()
    monitor.close()

    assert [update.status for update in updates].count(MonitorStatus.CLOSED) == 1
