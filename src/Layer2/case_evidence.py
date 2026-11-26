"""Build compact, case-local evidence for L3 RCA without using injection metadata."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "RCAEval_ds" / "RE2-OB"
DEFAULT_TOPOLOGY_PATH = (
    PROJECT_ROOT / "data" / "KnowledgeBase_processed" / "service_topology_0930.json"
)

_CASE_ID_RE = re.compile(r"^(?P<service>.+)_(?P<fault>cpu|delay|disk|loss|mem|socket)_(?P<run>\d+)$")
_LOG_PATTERNS = {
    "error_or_failure": re.compile(r"\berror\b|\bfail(?:ed|ure)?\b|exception", re.I),
    "timeout": re.compile(r"timeout|deadline exceeded", re.I),
    "connection": re.compile(r"connection|unavailable|refused|reset by peer", re.I),
    "memory": re.compile(r"oom|out.of.memory|memory|heap|allocation", re.I),
    "disk": re.compile(r"disk|i/o|fsync|storage", re.I),
    "network": re.compile(r"network|socket|packet|transmit|receive", re.I),
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _is_anomalous(candidate: dict) -> bool:
    if not candidate.get("zscore_valid"):
        return False
    duration = _number(candidate.get("duration_samples"))
    if duration is not None:
        return duration > 0
    start = _number(candidate.get("anomaly_start_timestamp"))
    end = _number(candidate.get("anomaly_end_timestamp"))
    return start is None or end is None or end > start


def select_service_by_coverage(candidates: list[dict], topology: dict) -> dict:
    """Select a service by own anomalous metrics plus rising caller latency/errors."""
    by_service: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        service = candidate.get("service")
        if service:
            by_service[service].append(candidate)
    if not by_service:
        return {"service": None, "coverage": {}, "tie_break_shap": None}

    coverage: dict[str, int] = {}
    for service, own_candidates in by_service.items():
        count = sum(_is_anomalous(candidate) for candidate in own_candidates)
        callers = topology.get(service, {}).get("upstream", [])
        for caller in callers:
            count += sum(
                _is_anomalous(candidate)
                and candidate.get("metric", "").lower() in {"latency", "error", "errors"}
                and str(candidate.get("direction", "")).lower() == "up"
                for candidate in by_service.get(caller, [])
            )
        coverage[service] = count

    highest_coverage = max(coverage.values())
    tied = [service for service, count in coverage.items() if count == highest_coverage]

    def service_shap(service: str) -> float:
        return max((_number(c.get("shap")) or 0.0) for c in by_service[service])

    selected = max(tied, key=service_shap)
    return {
        "service": selected,
        "coverage": coverage,
        "tie_break_shap": service_shap(selected),
    }


def _interval_for_service(candidates: list[dict], service: str) -> tuple[float, float] | None:
    intervals = []
    for candidate in candidates:
        if candidate.get("service") != service or not _is_anomalous(candidate):
            continue
        start = _number(candidate.get("anomaly_start_timestamp"))
        end = _number(candidate.get("anomaly_end_timestamp"))
        if start is not None and end is not None and end > start:
            intervals.append((start, end))
    if not intervals:
        return None
    # Use the union of this service's candidate anomaly intervals; never read inject_time.
    return min(start for start, _ in intervals), max(end for _, end in intervals)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_values(rows: list[dict], key: str, start: float, end: float) -> list[tuple[float, float]]:
    values = []
    for row in rows:
        timestamp = _number(row.get("time"))
        value = _number(row.get(key))
        if timestamp is not None and value is not None and start <= timestamp <= end:
            values.append((timestamp, value))
    values.sort(key=lambda item: item[0])
    return values


def _median(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    ordered = sorted(value for _, value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _counter_rate(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 2:
        return None
    increase = 0.0
    for (_, previous), (_, current) in zip(values, values[1:]):
        # Counter reset: after a reset, count the new counter value.
        increase += current - previous if current >= previous else current
    elapsed = values[-1][0] - values[0][0]
    return increase / elapsed if elapsed > 0 else None


def _ratio(before: float | None, during: float | None) -> float | None:
    if before is None or during is None or before == 0:
        return None
    return during / before


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def _resource_evidence(rows: list[dict], service: str, baseline: tuple[float, float],
                       anomaly: tuple[float, float]) -> dict:
    if not rows:
        return {"available": False}
    before_start, before_end = baseline
    anomaly_start, anomaly_end = anomaly

    def gauge(key: str) -> dict | None:
        if key not in rows[0]:
            return None
        before = _median(_selected_values(rows, key, before_start, before_end))
        during = _median(_selected_values(rows, key, anomaly_start, anomaly_end))
        if before is None and during is None:
            return None
        return {"baseline_median": _round(before, 2), "anomaly_median": _round(during, 2),
                "change_ratio": _round(_ratio(before, during), 3)}

    def counter(key: str) -> dict | None:
        if key not in rows[0]:
            return None
        before = _counter_rate(_selected_values(rows, key, before_start, before_end))
        during = _counter_rate(_selected_values(rows, key, anomaly_start, anomaly_end))
        if before is None and during is None:
            return None
        return {"baseline_rate_per_second": _round(before, 4),
                "anomaly_rate_per_second": _round(during, 4),
                "change_ratio": _round(_ratio(before, during), 3)}

    prefix = f"{service}_container-"
    signals = {
        "cpu_usage": counter(prefix + "cpu-usage-seconds-total"),
        "memory_working_set_bytes": gauge(prefix + "memory-working-set-bytes"),
        "memory_usage_bytes": gauge(prefix + "memory-usage-bytes"),
        "memory_limit_bytes": gauge(service + "_container-spec-memory-limit-bytes"),
        "sockets": gauge(prefix + "sockets"),
        "latency_p95": gauge(service + "_istio-latency-95"),
        "network_transmit_errors": counter(prefix + "network-transmit-errors-total"),
        "network_receive_errors": counter(prefix + "network-receive-errors-total"),
        "network_transmit_drops": counter(prefix + "network-transmit-packets-dropped-total"),
        "network_receive_drops": counter(prefix + "network-receive-packets-dropped-total"),
    }
    fs_keys = [key for key in rows[0] if key.startswith(prefix + "fs-") and key.endswith("bytes-total")]
    if fs_keys:
        signals["filesystem_io"] = {
            key.removeprefix(prefix): counter(key) for key in fs_keys
        }
    return {"available": True, "signals": {key: value for key, value in signals.items() if value is not None},
            "cpu_throttling": "unavailable: no throttling/CFS-throttled series in metrics.csv"}


def _canonical_service(name: str) -> str:
    return name[:-7] if name.endswith("service") else name


def _trace_evidence(rows: list[dict], services: list[str], baseline: tuple[float, float],
                    anomaly: tuple[float, float]) -> dict:
    if not rows:
        return {"available": False}
    target = {_canonical_service(service) for service in services}
    groups = defaultdict(lambda: {"baseline_durations_us": [], "anomaly_durations_us": [],
                                  "baseline_errors": 0, "anomaly_errors": 0})
    for row in rows:
        timestamp_ms = _number(row.get("startTimeMillis"))
        duration = _number(row.get("duration"))
        if timestamp_ms is None or duration is None:
            continue
        service = _canonical_service(row.get("serviceName", ""))
        if service not in target:
            continue
        timestamp = timestamp_ms / 1000
        window = "baseline" if baseline[0] <= timestamp < baseline[1] else (
            "anomaly" if anomaly[0] <= timestamp <= anomaly[1] else None
        )
        if window is None:
            continue
        groups[service][window + "_durations_us"].append(duration)
        status = (row.get("statusCode") or "").strip()
        if status not in {"", "0", "0.0"}:
            groups[service][window + "_errors"] += 1

    result = {}
    for service, values in groups.items():
        before = sorted(values["baseline_durations_us"])
        during = sorted(values["anomaly_durations_us"])
        def percentile95(items: list[float]) -> float | None:
            return items[min(len(items) - 1, int(0.95 * (len(items) - 1)))] if items else None
        before_p95, during_p95 = percentile95(before), percentile95(during)
        before_error_rate = values["baseline_errors"] / len(before) if before else None
        during_error_rate = values["anomaly_errors"] / len(during) if during else None
        result[service] = {
            "baseline_spans": len(before), "anomaly_spans": len(during),
            "baseline_p95_duration_us": _round(before_p95, 1),
            "anomaly_p95_duration_us": _round(during_p95, 1),
            "p95_change_ratio": _round(_ratio(before_p95, during_p95), 3),
            "baseline_error_spans": values["baseline_errors"],
            "anomaly_error_spans": values["anomaly_errors"],
            "baseline_error_rate": _round(before_error_rate, 4),
            "anomaly_error_rate": _round(during_error_rate, 4),
        }
    return {"available": True, "services": result}


def _redact(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<email>", text)
    text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b", "<id>", text)
    text = re.sub(r"\b[0-9a-fA-F]{24,}\b", "<id>", text)
    return re.sub(r"\s+", " ", text).strip()[:180]


def _log_evidence(rows: list[dict], services: list[str], baseline: tuple[float, float],
                  anomaly: tuple[float, float], max_samples: int = 3) -> dict:
    if not rows:
        return {"available": False}
    target = {_canonical_service(service) for service in services}
    counts = defaultdict(Counter)
    samples = []
    for row in rows:
        timestamp_ns = _number(row.get("timestamp"))
        if timestamp_ns is None:
            continue
        timestamp = timestamp_ns / 1e9
        service = _canonical_service(row.get("container_name", ""))
        if service not in target:
            continue
        window = "baseline" if baseline[0] <= timestamp < baseline[1] else (
            "anomaly" if anomaly[0] <= timestamp <= anomaly[1] else None
        )
        if window is None:
            continue
        level = (row.get("level") or "unknown").lower()
        message = " ".join((row.get("message") or "", row.get("error") or ""))
        counts[service][f"{window}_{level}"] += 1
        if level in {"error", "warning", "warn", "fatal", "critical"} or row.get("error"):
            for label, pattern in _LOG_PATTERNS.items():
                if pattern.search(message):
                    counts[service][f"{window}_{label}"] += 1
            if window == "anomaly" and len(samples) < max_samples:
                samples.append({"service": service, "level": level, "message": _redact(message)})
    return {"available": True, "counts_by_service": {s: dict(c) for s, c in counts.items()},
            "anomaly_samples": samples}


def build_case_evidence(report: dict, dataset_dir: Path | str = DEFAULT_DATASET_DIR,
                        topology_path: Path | str = DEFAULT_TOPOLOGY_PATH,
                        max_log_samples: int = 3) -> dict:
    """Return selected-service decision and compact evidence; does not read inject_time."""
    candidates = report.get("l2_candidates", [])
    try:
        topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        topology = {}
    selection = select_service_by_coverage(candidates, topology)
    selected = selection["service"]
    if not selected:
        return {"selected_service": None, "coverage": {}, "evidence": {"available": False}}

    intervals = _interval_for_service(candidates, selected)
    if intervals is None:
        return {"selected_service": selected, "coverage": selection["coverage"],
                "tie_break_shap": selection["tie_break_shap"],
                "evidence": {"available": False, "reason": "selected service has no candidate anomaly timestamps"}}

    start, end = intervals
    duration = max(end - start, 1.0)
    baseline = (start - duration, start)
    anomaly = (start, end)
    case_id = report.get("case_id") or report.get("timestamp") or ""
    match = _CASE_ID_RE.match(str(case_id))
    if not match:
        return {"selected_service": selected, "coverage": selection["coverage"],
                "tie_break_shap": selection["tie_break_shap"],
                "evidence": {"available": False, "reason": "case_id does not map to an RE2-OB case directory"}}

    case_dir = Path(dataset_dir) / f"{match.group('service')}_{match.group('fault')}" / match.group("run")
    upstream = topology.get(selected, {}).get("upstream", [])
    related_services = [selected, *[service for service in upstream if service != selected]]
    metrics_rows = _read_csv(case_dir / "metrics.csv")
    trace_rows = _read_csv(case_dir / "traces.csv")
    log_rows = _read_csv(case_dir / "logs.csv")

    return {
        "selected_service": selected,
        "coverage": selection["coverage"],
        "tie_break_shap": selection["tie_break_shap"],
        "candidate_names_in_selected_service": [
            candidate.get("name") for candidate in sorted(
                (c for c in candidates if c.get("service") == selected),
                key=lambda c: _number(c.get("shap")) or 0.0, reverse=True,
            )
        ],
        "evidence": {
            "available": bool(metrics_rows or trace_rows or log_rows),
            "time_window_source": "union of selected-service L2 anomaly intervals; no injection metadata",
            "baseline_window": [baseline[0], baseline[1]],
            "anomaly_window": [anomaly[0], anomaly[1]],
            "related_services": related_services,
            "resources": _resource_evidence(metrics_rows, selected, baseline, anomaly),
            "traces": _trace_evidence(trace_rows, related_services, baseline, anomaly),
            "logs": _log_evidence(log_rows, related_services, baseline, anomaly, max_log_samples),
        },
    }


def format_case_evidence(evidence: dict) -> str:
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
