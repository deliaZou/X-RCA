"""Scale-free effect sizes for Stage B resource and trace evidence.

Instead of raw change ratios (whose meaning depends on each metric's own
noise), compare the baseline-window samples with the anomaly-window samples of
the same metric and report:

* ``cliffs_delta``: P(anomaly > baseline) - P(anomaly < baseline), in [-1, 1].
  Non-parametric, robust to outliers, and still defined when the baseline is 0.
* ``robust_z``: (median_anomaly - median_baseline) / (1.4826 * MAD_baseline).
  Falls back to 1.2533 * mean absolute deviation when MAD is 0, and to ``None``
  when the baseline is perfectly constant. Not capped, so it can order
  evidence within one metric; it is not comparable across metric kinds.

Counters (cumulative series) are converted to per-second rates first, with the
same reset handling as ``case_evidence._counter_rate``.
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .case_evidence import _canonical_service, _number, _selected_values
except ImportError:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.rag_llm.case_evidence import _canonical_service, _number, _selected_values


MAD_SCALE = 1.4826
MEAN_AD_SCALE = 1.253314

# Same metric names as case_evidence._resource_evidence (prefix + suffix).
GAUGE_SUFFIXES = {
    "memory_working_set_bytes": "memory-working-set-bytes",
    "memory_usage_bytes": "memory-usage-bytes",
    "sockets": "sockets",
}
COUNTER_SUFFIXES = {
    "cpu_usage": "cpu-usage-seconds-total",
    "network_transmit_errors": "network-transmit-errors-total",
    "network_receive_errors": "network-receive-errors-total",
    "network_transmit_drops": "network-transmit-packets-dropped-total",
    "network_receive_drops": "network-receive-packets-dropped-total",
}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def cliffs_delta(baseline: list[float], anomaly: list[float]) -> float | None:
    """Cliff's delta in O((n + m) log m) using a sorted baseline."""
    if not baseline or not anomaly:
        return None
    ordered = sorted(baseline)
    size = len(ordered)
    greater = less = 0
    for value in anomaly:
        greater += bisect.bisect_left(ordered, value)        # baseline < value
        less += size - bisect.bisect_right(ordered, value)   # baseline > value
    return (greater - less) / (size * len(anomaly))


def robust_z(baseline: list[float], anomaly: list[float]) -> float | None:
    if not baseline or not anomaly:
        return None
    center = _median(baseline)
    scale = MAD_SCALE * _median([abs(v - center) for v in baseline])
    if scale == 0:
        scale = MEAN_AD_SCALE * (sum(abs(v - center) for v in baseline) / len(baseline))
    if scale == 0:
        return None
    return (_median(anomaly) - center) / scale


def _rates(values: list[tuple[float, float]]) -> list[float]:
    """Per-second increase of a cumulative counter, one value per sample interval.

    Exporters often refresh a counter only every few scrapes and the series is
    forward-filled in between. Differencing consecutive samples then yields a
    spike at each refresh and zeros elsewhere, so the rate distribution is
    dominated by zeros and hides real changes. Instead the rate is measured
    between *change points* (samples where the value changes) and assigned to
    every sample interval inside that span, i.e. the counter is interpolated
    linearly between refreshes. For a counter that changes at every sample this
    is identical to plain consecutive differencing. Intervals after the last
    refresh are dropped (their rate is unknown); a counter that never changes
    in the window has rate 0 everywhere. Counter resets are handled as before.
    """
    if len(values) < 2:
        return []
    rates: list[float] = []
    pending = 0                       # sample intervals since the last change point
    last_t, last_v = values[0]
    changed = False
    for t, v in values[1:]:
        if t <= last_t:
            continue
        pending += 1
        if v != last_v:
            changed = True
            rate = (v - last_v if v >= last_v else v) / (t - last_t)
            rates.extend([rate] * pending)
            pending = 0
            last_t, last_v = t, v
    if not changed:
        return [0.0] * pending
    return rates


def _effect(baseline: list[float], anomaly: list[float]) -> dict[str, Any]:
    delta, z = cliffs_delta(baseline, anomaly), robust_z(baseline, anomaly)
    return {"n_baseline": len(baseline), "n_anomaly": len(anomaly),
            "cliffs_delta": round(delta, 4) if delta is not None else None,
            "robust_z": round(z, 3) if z is not None else None}


def resource_effects(rows: list[dict], service: str, baseline: tuple[float, float],
                     anomaly: tuple[float, float]) -> dict[str, Any]:
    """Effect size of every resource signal of ``service``: baseline vs anomaly window."""
    if not rows:
        return {"available": False}
    prefix = f"{service}_container-"
    signals: dict[str, Any] = {}
    for name, suffix in GAUGE_SUFFIXES.items():
        key = prefix + suffix
        if key in rows[0]:
            before = [v for _, v in _selected_values(rows, key, *baseline)]
            during = [v for _, v in _selected_values(rows, key, *anomaly)]
            if before and during:
                signals[name] = _effect(before, during)
    for name, suffix in COUNTER_SUFFIXES.items():
        key = prefix + suffix
        if key in rows[0]:
            before = _rates(_selected_values(rows, key, *baseline))
            during = _rates(_selected_values(rows, key, *anomaly))
            if before and during:
                signals[name] = _effect(before, during)
    # Filesystem I/O counters are emitted with device-specific names. Compare
    # each rate series independently so the effect size remains scale-free.
    for key in rows[0]:
        if not (key.startswith(prefix + "fs-") and key.endswith("bytes-total")):
            continue
        name = key.removeprefix(prefix)
        before = _rates(_selected_values(rows, key, *baseline))
        during = _rates(_selected_values(rows, key, *anomaly))
        if before and during:
            signals[name] = _effect(before, during)
    return {"available": True, "method": "cliffs_delta + robust_z (baseline vs anomaly samples)",
            "signals": signals}


def trace_effects(rows: list[dict], services: list[str], baseline: tuple[float, float],
                  anomaly: tuple[float, float]) -> dict[str, Any]:
    """Cliff's delta of span durations, anomaly window vs baseline window, per service.

    Signed: a positive value means anomaly-window spans tend to be slower than
    baseline spans. Spans of a service are pooled over operations, exactly as the
    p95 change ratio this replaces. Service matching and window boundaries mirror
    ``case_evidence._trace_evidence`` so the span counts agree with it.
    """
    if not rows:
        return {"available": False}
    target = {_canonical_service(service) for service in services}
    durations: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "anomaly": []})
    for row in rows:
        timestamp_ms = _number(row.get("startTimeMillis"))
        duration = _number(row.get("duration"))
        if timestamp_ms is None or duration is None:
            continue
        service = _canonical_service(row.get("serviceName", ""))
        if service not in target:
            continue
        timestamp = timestamp_ms / 1000
        if baseline[0] <= timestamp < baseline[1]:
            durations[service]["baseline"].append(duration)
        elif anomaly[0] <= timestamp <= anomaly[1]:
            durations[service]["anomaly"].append(duration)

    result = {}
    for service, values in durations.items():
        delta = cliffs_delta(values["baseline"], values["anomaly"])
        result[service] = {"n_baseline": len(values["baseline"]),
                           "n_anomaly": len(values["anomaly"]),
                           "cliffs_delta": round(delta, 4) if delta is not None else None}
    return {"available": True,
            "method": "cliffs_delta of span durations (anomaly vs baseline window, signed)",
            "services": result}
