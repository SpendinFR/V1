"""Lightweight system monitoring utilities used for physiological drives.

This module exposes a :class:`SystemMonitor` helper that samples the host
machine to gather CPU, memory, GPU and disk metrics.  The collected snapshot is
consumed by the orchestrator to translate hardware pressure into homeostatic
needs (energy, respiration, thermoregulation, etc.).

The implementation is intentionally defensive: every probe is wrapped so that a
missing sensor (e.g. no NVIDIA GPU) or insufficient permissions simply results
in a ``None`` value instead of an exception bubbling up to the caller.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import psutil

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _bytes_to_human(n: float) -> float:
    """Convert a raw byte/sec throughput to megabytes for easier digestion."""

    return float(n) / (1024.0 * 1024.0)


@dataclass
class DiskSnapshot:
    """Disk I/O snapshot used to compute per-second metrics."""

    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int


@dataclass
class SystemMonitor:
    """Polls the local machine and exposes normalised metrics."""

    interval: float = 2.0
    _last_poll: float = field(default=0.0, init=False)
    _last_snapshot: Dict[str, Any] = field(default_factory=dict, init=False)
    _prev_disk: Dict[str, DiskSnapshot] = field(default_factory=dict, init=False)
    _prev_disk_ts: float = field(default=0.0, init=False)

    def poll(self, force: bool = False) -> Dict[str, Any]:
        """Return a fresh snapshot of system metrics.

        The snapshot closely mirrors the structure provided by the user's
        original monitoring script so that downstream components can reuse the
        values with minimal adaptation.
        """

        now = time.time()
        if not force and self._last_snapshot and (now - self._last_poll) < max(0.5, self.interval):
            return dict(self._last_snapshot)

        snapshot: Dict[str, Any] = {"timestamp": now}

        cpu_load: Optional[float] = None
        cpu_temp: Optional[float] = None
        try:
            cpu_load = float(psutil.cpu_percent(interval=0.05))
            temps = getattr(psutil, "sensors_temperatures", None)
            if temps:
                data = temps()
                if isinstance(data, dict):
                    # Grab the first available CPU temperature sensor.
                    for _, entries in data.items():
                        for entry in entries:
                            label = getattr(entry, "label", "") or ""
                            if "cpu" in label.lower() or "package" in label.lower() or not label:
                                cpu_temp = float(getattr(entry, "current", None) or 0.0) or cpu_temp
                                if cpu_temp:
                                    break
                        if cpu_temp:
                            break
        except Exception:
            cpu_load = cpu_load or None
            cpu_temp = None

        snapshot["cpu"] = {"load": cpu_load, "temp_c": cpu_temp}

        try:
            mem = psutil.virtual_memory()
            snapshot["memory"] = {
                "used_gb": float(mem.used) / (1024 ** 3),
                "total_gb": float(mem.total) / (1024 ** 3),
                "percent": float(mem.percent),
            }
        except Exception:
            snapshot["memory"] = {}

        snapshot["gpu"] = self._query_gpu()
        snapshot["power"] = {}
        if snapshot["gpu"] and snapshot["gpu"].get("power_w") is not None:
            snapshot["power"]["draw_w"] = snapshot["gpu"].get("power_w")

        snapshot["disks"] = self._query_disks(now)

        insights = self._llm_comment(snapshot)
        if insights:
            snapshot["llm_analysis"] = insights
        self._last_snapshot = snapshot
        self._last_poll = now
        return dict(snapshot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _query_gpu(self) -> Dict[str, Any]:
        fields = [
            "temperature.gpu",
            "fan.speed",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "power.draw",
        ]
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2)
        except Exception:
            return {}

        row = [val.strip() for val in output.strip().split(",")]
        if len(row) < 6:
            return {}

        try:
            temp_c = float(row[0])
        except Exception:
            temp_c = None
        try:
            fan_pct = float(row[1])
        except Exception:
            fan_pct = None
        try:
            util_pct = float(row[2])
        except Exception:
            util_pct = None
        try:
            mem_used = float(row[3])
        except Exception:
            mem_used = None
        try:
            mem_total = float(row[4])
        except Exception:
            mem_total = None
        try:
            power = float(row[5])
        except Exception:
            power = None

        payload = {
            "temp_c": temp_c,
            "fan_pct": fan_pct,
            "util_pct": util_pct,
            "mem_used_mb": mem_used,
            "mem_total_mb": mem_total,
            "power_w": power,
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _query_disks(self, now: float) -> Iterable[Dict[str, Any]]:
        try:
            counters = psutil.disk_io_counters(perdisk=True)
        except Exception:
            return []

        disks = []
        prev_ts = self._prev_disk_ts or now
        dt = max(0.001, now - prev_ts)
        for name, stats in counters.items():
            prev = self._prev_disk.get(name)
            if prev is None:
                self._prev_disk[name] = DiskSnapshot(
                    read_bytes=stats.read_bytes,
                    write_bytes=stats.write_bytes,
                    read_count=stats.read_count,
                    write_count=stats.write_count,
                )
                continue

            read_rate = max(0.0, stats.read_bytes - prev.read_bytes) / dt
            write_rate = max(0.0, stats.write_bytes - prev.write_bytes) / dt
            read_iops = max(0.0, stats.read_count - prev.read_count) / dt
            write_iops = max(0.0, stats.write_count - prev.write_count) / dt

            current_snapshot = DiskSnapshot(
                read_bytes=stats.read_bytes,
                write_bytes=stats.write_bytes,
                read_count=stats.read_count,
                write_count=stats.write_count,
            )
            self._prev_disk[name] = current_snapshot

            if read_rate + write_rate < 1024.0 and read_iops + write_iops < 0.1:
                continue

            disks.append(
                {
                    "name": name,
                    "read_mb_s": _bytes_to_human(read_rate),
                    "write_mb_s": _bytes_to_human(write_rate),
                    "read_iops": read_iops,
                    "write_iops": write_iops,
                }
            )

        self._prev_disk_ts = now
        return disks

    def _llm_comment(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = try_call_llm_dict(
            "system_monitor",
            input_payload=snapshot,
            logger=LOGGER,
        )
        if not response:
            return None
        observations = response.get("observations")
        if observations is not None and not isinstance(observations, list):
            observations = None
        risks = response.get("risks")
        if risks is not None and not isinstance(risks, list):
            risks = None
        return {
            "observations": observations or [],
            "risks": risks or [],
            "notes": response.get("notes", ""),
        }

