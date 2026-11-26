"""Stage A: rerank L2 metric candidates and prepare minimal Stage B evidence."""

from __future__ import annotations

import re
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


_SERVICE_SUFFIX = re.compile(r"service$", re.IGNORECASE)
_SERVICE_CLASS = re.compile(r"(?:^|\.)([A-Za-z][A-Za-z0-9]*)Service(?:/|$)", re.IGNORECASE)


def _service_key(value: Any) -> str:
    """Normalize service labels for comparisons, retaining original output labels."""
    text = str(value or "").strip().lower()
    return _SERVICE_SUFFIX.sub("", text)


def _is_frontend(value: Any) -> bool:
    return _service_key(value) == "frontend"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number != float("inf") else None


def _operation(row: Mapping[str, Any]) -> str:
    return str(row.get("operation") or row.get("operationName") or row.get("name") or "")


def _caller(row: Mapping[str, Any]) -> str:
    return str(row.get("caller") or row.get("service") or row.get("serviceName") or "").strip()


def _target(operation: str) -> str | None:
    """Resolve a target from e.g. ``hipstershop.CurrencyService/Convert``."""
    match = _SERVICE_CLASS.search(operation)
    if match:
        return _service_key(match.group(1))
    # Also support compact names such as ``currencyservice/Convert``.
    head = operation.split("/", 1)[0].rsplit(".", 1)[-1]
    head = _service_key(head)
    return head or None


def _ratio(row: Mapping[str, Any]) -> float | None:
    for key in ("ratio", "fault_to_pre_ratio", "fault_vs_pre", "vs_pre_3m", "multiple"):
        value = _number(row.get(key))
        if value is not None:
            return value
    fault = _number(row.get("fault_duration") or row.get("anomaly_duration") or row.get("fault_median"))
    pre = _number(row.get("pre_duration") or row.get("baseline_duration") or row.get("pre_median"))
    return fault / pre if fault is not None and pre is not None else None


def run_stage_a(
    service_coverage: Mapping[str, Any] | None,
    l2_candidates: Sequence[Mapping[str, Any]] | None,
    slow_operations: Sequence[Mapping[str, Any]] | None,
    *,
    egress_threshold: float = 30.0,
    multi_caller_threshold: float = 5.0,
) -> dict[str, Any]:
    """Rank candidate services by E > M > coverage, then rank their metrics.

    Slow-operation rows should describe client spans. Accepted fields include
    ``caller``, ``operation`` and ``ratio``; alternatively supply fault/pre
    duration values (``fault_duration`` and ``pre_duration``). A ratio is
    interpreted as fault-window median divided by pre-window median.
    """
    coverage = service_coverage or {}
    candidates = list(l2_candidates or [])
    operations = list(slow_operations or [])

    # Candidate services are only those represented in the L2 top-five metrics.
    service_names: dict[str, str] = {}
    for name in (c.get("service") for c in candidates):
        if name is None or not str(name).strip():
            continue
        service_names.setdefault(_service_key(name), str(name).strip())

    slow: list[dict[str, Any]] = []
    for row in operations:
        caller, operation, ratio = _caller(row), _operation(row), _ratio(row)
        target = _target(operation)
        if caller and target and ratio is not None and ratio >= multi_caller_threshold:
            slow.append({"caller": caller, "target": target, "operation": operation, "multiple": ratio})

    # Rule E: an instrumented service's own TraceService/Export client span.
    candidate_keys = set(service_names)
    e_hits = [item for item in slow
              if not _is_frontend(item["caller"])
              and _service_key(item["caller"]) in candidate_keys
              and item["multiple"] >= egress_threshold
              and "traceservice/export" in item["operation"].lower()]
    e_by_service: dict[str, dict[str, Any]] = {}
    for item in e_hits:
        key = _service_key(item["caller"])
        if key not in e_by_service or item["multiple"] > e_by_service[key]["multiple"]:
            e_by_service[key] = item

    # Rule M: a non-frontend target is slow for at least two distinct callers.
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in slow:
        if not _is_frontend(item["target"]):
            by_target[item["target"]].append(item)
    m_by_service: dict[str, dict[str, Any]] = {}
    for target, rows in by_target.items():
        distinct_callers = {_service_key(row["caller"]) for row in rows if not _is_frontend(row["caller"])}
        if len(distinct_callers) >= 2 and target in candidate_keys:
            row = {"callers": sorted(distinct_callers),
                   "multiple": max(item["multiple"] for item in rows)}
            if target not in m_by_service or row["multiple"] > m_by_service[target]["multiple"]:
                m_by_service[target] = row

    shap_by_service = {
        key: max((float(item.get("shap") or 0) for item in candidates
                  if _service_key(item.get("service")) == key), default=0.0)
        for key in service_names
    }
    service_rows = []
    for key, service in service_names.items():
        if key in e_by_service:
            tier, source, support = 0, "egress_network", e_by_service[key]["multiple"]
            reason = {"rule": "E", "operation": e_by_service[key]["operation"],
                      "multiple": support, "threshold": egress_threshold}
        elif key in m_by_service:
            tier, source, support = 1, "multi_caller", m_by_service[key]["multiple"]
            reason = {"rule": "M", "callers": m_by_service[key]["callers"],
                      "multiple": support, "threshold": multi_caller_threshold}
        else:
            tier, source, support = 2, "coverage", float(coverage.get(service, coverage.get(key, 0)) or 0)
            reason = {"rule": "coverage", "coverage_score": support}
        service_rows.append({"_key": key, "service": service, "_tier": tier, "source": source,
                             "support_score": support, "coverage_score": coverage.get(service, coverage.get(key, 0)),
                             "shap_tiebreak": shap_by_service[key], "reason": reason})

    service_rows.sort(key=lambda row: (row["_tier"], -float(row["support_score"] or 0),
                                       -float(row["coverage_score"] or 0),
                                       -row["shap_tiebreak"], row["service"].lower()))
    ranked_services = []
    service_rank_by_key = {}
    for rank, row in enumerate(service_rows, 1):
        service_rank_by_key[row["_key"]] = rank
        ranked_services.append({"rank": rank, "service": row["service"], "source": row["source"],
                                "coverage_score": row["coverage_score"],
                                "support_score": row["support_score"], "reason": row["reason"]})

    metrics_by_key: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for l2_rank, item in enumerate(candidates, 1):
        key = _service_key(item.get("service"))
        metrics_by_key[key].append((l2_rank, item))
    ranked_metrics = []
    for key in service_rank_by_key:
        group = sorted(metrics_by_key[key],
                       key=lambda pair: (-float(pair[1].get("shap") or 0), pair[0]))
        for l2_rank, item in group:
            ranked_metrics.append({"rank": len(ranked_metrics) + 1,
                                   "l2_rank": l2_rank,
                                   "service_rank": service_rank_by_key[key],
                                   "service": service_names[key],
                                   "name": item.get("name"),
                                   "metric": item.get("metric"),
                                   "shap": item.get("shap"),
                                   "direction": item.get("direction")})

    top_service = ranked_services[0]["service"] if ranked_services else None
    top_key = _service_key(top_service)
    top_metrics = [row for row in ranked_metrics if _service_key(row["service"]) == top_key]
    stage_b_input = {"should_run": len(top_metrics) > 1, "service": top_service}
    if len(top_metrics) > 1:
        stage_b_input["candidate_metrics"] = [row["name"] for row in top_metrics]
    return {
        "ranked_services": ranked_services,
        "ranked_metrics": ranked_metrics,
        "stage_b_input": stage_b_input,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return list(csv.DictReader(stream))


def _client_slow_operations(trace_rows: list[dict[str, str]], window: tuple[float, float],
                            baseline_seconds: float = 180.0) -> list[dict[str, Any]]:
    """Calculate per-caller operation median ratios from traces.csv."""
    start, end = window
    before: dict[tuple[str, str], list[float]] = defaultdict(list)
    during: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in trace_rows:
        timestamp = _number(row.get("startTimeMillis"))
        duration = _number(row.get("duration"))
        caller = (row.get("serviceName") or "").strip()
        operation = (row.get("operationName") or row.get("methodName") or "").strip()
        if timestamp is None or duration is None or not caller or not operation:
            continue
        timestamp /= 1000
        key = (caller, operation)
        if start - baseline_seconds <= timestamp < start:
            before[key].append(duration)
        elif start <= timestamp <= end:
            during[key].append(duration)

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    result = []
    for key, fault_values in during.items():
        pre_values = before.get(key)
        if not pre_values:
            continue
        pre, fault = median(pre_values), median(fault_values)
        if pre <= 0:
            continue
        result.append({"caller": key[0], "operation": key[1], "pre_duration": pre,
                       "fault_duration": fault, "ratio": fault / pre,
                       "pre_spans": len(pre_values), "fault_spans": len(fault_values)})
    return result


def build_stage_a_inputs(l2_output: Mapping[str, Any], dataset_dir: Path,
                         topology_path: Path) -> dict[str, Any]:
    """Load only L2 top-five metrics and traces needed for Stage A ranking."""
    # Imports are local so the rule module remains importable as a standalone utility.
    try:
        from .case_evidence import (
            _interval_for_service, select_service_by_coverage,
        )
        from .l2_adapter import convert_l2_to_xai_report
    except ImportError:  # direct execution: python src/Layer2/stage_a_1014.py
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.Layer2.case_evidence import (
            _interval_for_service, select_service_by_coverage,
        )
        from src.Layer2.l2_adapter import convert_l2_to_xai_report

    report = convert_l2_to_xai_report(
        dict(l2_output), candidate_pool_size=max(len(l2_output.get("metrics_list", [])), 1)
    )
    all_candidates = report.get("l2_candidates", [])
    # Stage A candidates are the first five SHAP-ranked metric entries.
    top_metrics = all_candidates[:5]
    # Coverage is computed from the same top-five metric candidates, not every
    # lower-ranked metric belonging to those services.
    candidates = top_metrics
    if not candidates:
        raise ValueError("L2 输出中没有可用的前 5 候选服务及其指标候选。")
    topology = json.loads(topology_path.read_text(encoding="utf-8")) if topology_path.exists() else {}
    coverage_selection = select_service_by_coverage(candidates, topology)
    s0 = coverage_selection.get("service")
    if not s0:
        raise ValueError("覆盖法没有选出服务；请检查 L2 输出中的候选与异常信息。")

    # Prefer the detector-generated PCA/AD window from the raw L2 JSON.
    # Fall back to metric intervals only for older L2 files without this field.
    ad_window = l2_output.get("selected_ad_window") or {}
    start = _number(ad_window.get("start_timestamp"))
    end = _number(ad_window.get("end_timestamp"))
    if start is not None and end is not None and end >= start:
        time_window_source = "selected_ad_window"
    else:
        interval = _interval_for_service(candidates, s0)
        if interval is None:
            raise ValueError(
                f"L2 输出没有有效的 selected_ad_window，且 {s0} 没有可用的指标时间窗。"
            )
        start, end = interval
        time_window_source = "metric_interval_fallback"
    duration = max(end - start, 1.0)
    baseline, anomaly = (start - duration, start), (start, end)

    case_id = (l2_output.get("_meta") or {}).get("case_id") or report.get("case_id")
    match = re.match(r"^(?P<service>.+)_(?P<fault>cpu|delay|disk|loss|mem|socket)_(?P<run>\d+)$",
                     str(case_id or ""), re.IGNORECASE)
    if not match:
        raise ValueError(f"L2 输出 case_id 无法映射到数据目录：{case_id!r}")
    case_dir = dataset_dir / f"{match.group('service')}_{match.group('fault').lower()}" / match.group("run")
    traces = _read_csv(case_dir / "traces.csv")

    return {
        "service_coverage": coverage_selection.get("coverage", {}),
        "l2_candidates": candidates,
        "slow_operations": _client_slow_operations(traces, anomaly),
        "_trace_rows": traces,
        "case_id": case_id,
        "data_directory": str(case_dir),
        "time_window_source": time_window_source,
        "baseline_window": baseline,
        "anomaly_window": anomaly,
    }


def _build_stage_b_evidence(case_dir: Path, service: str,
                            baseline: tuple[float, float], anomaly: tuple[float, float],
                            trace_rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Build the minimal Stage B evidence for the top-ranked service; logs are omitted."""
    try:
        from .case_evidence import _resource_evidence, _trace_evidence
        from .effect_size import resource_effects          # 新增

    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.Layer2.case_evidence import _resource_evidence, _trace_evidence
        from src.Layer2.effect_size import resource_effects   # 新增


    metrics = _read_csv(case_dir / "metrics.csv")
    traces = trace_rows if trace_rows is not None else _read_csv(case_dir / "traces.csv")
    return {
        "baseline_window": list(baseline),
        "anomaly_window": list(anomaly),
        "resources": _resource_evidence(metrics, service, baseline, anomaly),
        "resource_effects": resource_effects(metrics, service, baseline, anomaly),  # 新增
        "traces": _trace_evidence(traces, [service], baseline, anomaly),
    }


def main() -> int:
    """Run one L2 JSON or every JSON in a folder; configure paths below."""
    project_root = Path(__file__).resolve().parents[2]
    # Set L2_INPUT to one JSON file for a single case, or a directory for batch mode.
    # l2_input = project_root / "src" / "pcc_kernelSHAP" / "layer2_output_time" / "checkoutservice_cpu_2.json"

    # 批量运行时改成目录
    l2_input = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer1_output"
    dataset_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB"
    topology_path = project_root / "Dataset" / "RCAEval_ds" / "service_topology.json"
    output_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageA_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if l2_input.is_file():
        l2_paths = [l2_input]
    elif l2_input.is_dir():
        l2_paths = sorted(l2_input.glob("*.json"))
    else:
        raise FileNotFoundError(f"L2 输入路径不存在：{l2_input}")
    if not l2_paths:
        raise FileNotFoundError(f"L2 输入目录中没有 JSON 文件：{l2_input}")

    case_results = []
    failures = []
    for l2_path in l2_paths:
        try:
            l2_output = json.loads(l2_path.read_text(encoding="utf-8"))
            inputs = build_stage_a_inputs(l2_output, dataset_dir, topology_path)
            result = run_stage_a(**{key: inputs[key] for key in (
                "service_coverage", "l2_candidates", "slow_operations")})
            stage_b = result["stage_b_input"]
            if stage_b["should_run"] and stage_b["service"]:
                stage_b["evidence"] = _build_stage_b_evidence(
                    Path(inputs["data_directory"]), stage_b["service"],
                    inputs["baseline_window"], inputs["anomaly_window"], inputs["_trace_rows"])
            ground_truth = ((l2_output.get("_meta") or {}).get("ground_truth") or {}).get("service")
            top_service = stage_b["service"]
            hit = bool(ground_truth) and _service_key(top_service) == _service_key(ground_truth)
            result["case_id"] = inputs["case_id"]
            result["time_window_source"] = inputs["time_window_source"]
            case_results.append({"result": result, "ground_truth_service": ground_truth,
                                 "coarse_hit": hit})
            case_path = output_dir / f"{inputs['case_id']}_stage_a.json"
            case_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{len(case_results)}] {inputs['case_id']}: top_service={top_service}, "
                  f"truth={ground_truth}, hit={hit}")
        except Exception as exc:
            failures.append({"l2_file": l2_path.name, "error": str(exc)})
            print(f"[ERROR] {l2_path.name}: {exc}")

    evaluated = [case for case in case_results if case.get("ground_truth_service")]
    hits = sum(bool(case["coarse_hit"]) for case in evaluated)
    summary = {
        "metric": "coarse_service_exact_match",
        "definition": "top-ranked service equals _meta.ground_truth.service after removing a trailing 'service' suffix",
        "total_l2_files": len(l2_paths),
        "successful_cases": len(case_results),
        "evaluated_cases_with_ground_truth": len(evaluated),
        "hits": hits,
        "coarse_hit_rate": hits / len(evaluated) if evaluated else None,
        "failures": failures,
        "cases": [{"case_id": case["result"].get("case_id"),
                   "top_ranked_service": (case["result"].get("stage_b_input") or {}).get("service"),
                   "ground_truth_service": case.get("ground_truth_service"),
                   "source": ((case["result"].get("ranked_services") or [{}])[0]).get("source"),
                   "coarse_hit": case.get("coarse_hit")} for case in case_results],
    }
    summary_name = (f"{l2_input.stem}_coarse_hit_rate_summary.json"
                    if l2_input.is_file() else "coarse_hit_rate_summary.json")
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nCoarse 命中率：{hits}/{len(evaluated)} = "
          f"{summary['coarse_hit_rate']:.2%}" if evaluated else "\n没有可评估的 ground truth 案例。")
    print(f"逐案结果及汇总已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
