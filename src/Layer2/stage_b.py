"""Stage B: evidence-weighted metric reranking for Stage A's top service.

Stage B does not choose a service or declare a root cause. It only reorders
the metrics belonging to Stage A's first-ranked service and leaves the
remaining Stage A metric order intact.

Trace evidence follows the fault semantics of RCAEval (Sec. 3.2):
  * Loss  (error spans present + p95 latency rise): the root-cause indicator
    is the failed-request / error metric, so ``error`` metrics get the
    strongest weight; latency is only a shared symptom and gets a lower one.
  * Delay (no error spans + p95 latency rise): the root-cause indicator is
    the latency metric, so only latency metrics are weighted.

Resource evidence (cpu / memory / network) is independent of the Trace branch
and uses a robust z-score of the anomaly window against the baseline window
(effect_size.robust_z: median / 1.4826*MAD of the baseline) plus a minimum
practical effect. A signal counts as evidence only if BOTH hold:
  1. significance : robust z >= Z_GATE  (positive direction only);
  2. practical    : change_ratio >= 1 + MIN_RELATIVE_EFFECT.
RCAEval resource faults are stress injections, so only increases count.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MIN_SPAN_COUNT = 20
LOSS_LATENCY_RATIO = 1.30
DELAY_LATENCY_RATIO = 1.30

# --- resource evidence: robust z + minimum effect ---------------------------------
# 3.5 is the conventional modified-z outlier cut-off (Iglewicz & Hoaglin 1993 --
# verify the citation); a statistical convention, not tuned on this dataset.
Z_GATE = 3.5
# Minimum practical effect: the signal must also rise by at least this fraction
# of its baseline level. A user-set floor against statistically "significant"
# but operationally negligible drift (slow gauges have a tiny baseline MAD).
# Report it as a sensitivity parameter (try 0.05 / 0.10 / 0.20).
MIN_RELATIVE_EFFECT = 0.10
MIN_EFFECT_SAMPLES = 20
RESOURCE_SCORE = 3
# Optional (OFF): order tied resource candidates by change_ratio instead of SHAP.
# Tested on RE2-OB and rejected: memory-stress faults also drive cpu 36-68x
# (injected memory only 3-25x), so cross-kind change_ratio picks cpu over the true
# memory metric (fine AC@1 54/90 -> 51/90). Kept only for the ablation.
ORDER_RESOURCE_TIES_BY_CHANGE = False

KIND_SIGNAL_KEYS = {
    "cpu": ("cpu_usage",),
    "memory": ("memory_working_set_bytes", "memory_usage_bytes"),
    "network": ("sockets", "network_transmit_errors", "network_receive_errors",
                "network_transmit_drops", "network_receive_drops"),
}

# Trace-branch weights (score 3 = strong, 2 = medium, 1 = weak).
LOSS_ERROR_SCORE = 3     # Loss: error metric is the root-cause indicator.
LOSS_LATENCY_SCORE = 2   # Loss: latency is a shared symptom, below error.
DELAY_LATENCY_SCORE = 3  # Delay: latency metric is the root-cause indicator.

_LEVELS = {3: "strong", 2: "medium", 1: "weak", 0: "none"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_service(value: Any) -> str:
    return re.sub(r"service$", "", str(value or "").strip().lower())


def _metric_kind(row: Mapping[str, Any]) -> str:
    raw = str(row.get("metric") or row.get("name") or "").lower()
    raw = raw.rsplit("_", 1)[-1]
    if "latency" in raw or raw in {"loss", "packet_loss", "network_loss"}:
        return "loss_delay"
    if "error" in raw:
        return "error"
    if "disk" in raw or "filesystem" in raw or "fs_" in raw:
        return "disk"
    if "mem" in raw or "memory" in raw:
        return "memory"
    if "cpu" in raw or "processor" in raw:
        return "cpu"
    if "socket" in raw or "network" in raw or "packet" in raw:
        return "network"
    return raw


def _trace_record(evidence: Mapping[str, Any], service: str) -> Mapping[str, Any]:
    traces = evidence.get("traces") or {}
    services = traces.get("services") or {}
    key = _canonical_service(service)
    for name, record in services.items():
        if _canonical_service(name) == key:
            return record if isinstance(record, Mapping) else {}
    return {}


def _filesystem_io_anomaly(evidence: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    resources = evidence.get("resources") or {}
    signals = resources.get("signals") or {}
    filesystem = signals.get("filesystem_io") or {}
    positive = []
    for metric_name, values in filesystem.items():
        if not isinstance(values, Mapping):
            continue
        rate = _number(values.get("anomaly_rate_per_second"))
        if rate is not None and rate > 0:
            positive.append({"metric": metric_name, "anomaly_rate_per_second": rate})
    return bool(positive), positive


def _resource_effect(evidence: Mapping[str, Any], kind: str, z_gate: float,
                     min_relative_effect: float) -> tuple[dict[str, Any] | None, list[str]]:
    """Best signal of ``kind`` passing both gates, plus why the others did not.

    Returns (best, notes). ``best`` has the largest robust z among passing signals;
    ``notes`` explains every signal that was missing or rejected (so a missing
    cpu_usage shows up in the evidence text instead of silently scoring 0).
    """
    effects = (evidence.get("resource_effects") or {}).get("signals") or {}
    changes = (evidence.get("resources") or {}).get("signals") or {}
    best: dict[str, Any] | None = None
    notes: list[str] = []
    for key in KIND_SIGNAL_KEYS.get(kind, ()):
        item = effects.get(key)
        if not item:
            notes.append(f"{key}：无效应量记录（缺列或窗口内无样本）。")
            continue
        n = int(min(_number(item.get("n_baseline")) or 0, _number(item.get("n_anomaly")) or 0))
        if n < MIN_EFFECT_SAMPLES:
            notes.append(f"{key}：样本不足 (n={n} < {MIN_EFFECT_SAMPLES})。")
            continue
        z = _number(item.get("robust_z"))
        if z is None:
            notes.append(f"{key}：基线无波动，robust z 不可定义。")
            continue
        ratio = _number((changes.get(key) or {}).get("change_ratio"))
        if z < z_gate:
            notes.append(f"{key}：robust z={z:.2f} < {z_gate}（未达显著或方向为下降）。")
            continue
        if ratio is None or ratio < 1.0 + min_relative_effect:
            shown = "NA" if ratio is None else f"{ratio:.3f}"
            notes.append(f"{key}：robust z={z:.2f} 达标，但 change_ratio={shown} "
                         f"< {1.0 + min_relative_effect:.2f}，未达最小效应。")
            continue
        if best is None or z > best["robust_z"]:
            best = {"signal": key, "robust_z": z, "change_ratio": ratio}
    return best, notes


def _loss_triggered(spans_sufficient: bool, has_error_spans: bool,
                    latency_ratio: float | None) -> bool:
    return bool(spans_sufficient and has_error_spans and latency_ratio is not None
                and latency_ratio >= LOSS_LATENCY_RATIO)


def _delay_triggered(spans_sufficient: bool, has_error_spans: bool,
                     latency_ratio: float | None) -> bool:
    return bool(spans_sufficient and not has_error_spans and latency_ratio is not None
                and latency_ratio >= DELAY_LATENCY_RATIO)


def _weight_for_metric(
    row: Mapping[str, Any],
    *,
    trace: Mapping[str, Any],
    latency_ratio: float | None,
    has_error_spans: bool,
    spans_sufficient: bool,
    disk_anomaly: bool,
    disk_signals: list[dict[str, Any]],
    evidence: Mapping[str, Any],
    z_gate: float,
    min_relative_effect: float,
) -> dict[str, Any]:
    kind = _metric_kind(row)
    score = 0
    reasons: list[str] = []
    effect_z: float | None = None
    effect_ratio: float | None = None

    if _loss_triggered(spans_sufficient, has_error_spans, latency_ratio):
        # Loss: the failed-request / error metric is the root-cause indicator.
        if kind == "error":
            score = LOSS_ERROR_SCORE
            reasons.append("Loss：故障窗存在 Trace error 且 p95 延迟上升达到阈值，"
                           "error 类指标为 Loss 的根因指标。")
        elif kind == "loss_delay":
            score = LOSS_LATENCY_SCORE
            reasons.append("Loss：延迟上升是共有症状，权重低于 error 类指标。")
    elif _delay_triggered(spans_sufficient, has_error_spans, latency_ratio):
        # Delay: only the latency metric is the root-cause indicator.
        if kind == "loss_delay":
            score = DELAY_LATENCY_SCORE
            reasons.append("Delay：故障窗无 Trace error 且 p95 延迟上升达到阈值，"
                           "延迟指标为 Delay 的根因指标。")

    if kind == "disk":
        # Disk candidates receive no weight without observed filesystem I/O.
        if disk_anomaly:
            score = 3
            reasons.append("故障窗至少一个 filesystem I/O anomaly rate 大于 0。")
            reasons.extend(
                f"{item['metric']} anomaly_rate_per_second={item['anomaly_rate_per_second']}"
                for item in disk_signals
            )
        else:
            score = 0
            reasons.append("未观测到 filesystem I/O；disk/diskio 候选不加权。")
    elif kind in {"cpu", "memory", "network"}:
        effect, notes = _resource_effect(evidence, kind, z_gate, min_relative_effect)
        if effect is not None:
            effect_z = effect["robust_z"]
            effect_ratio = effect["change_ratio"]
            score = max(score, RESOURCE_SCORE)
            reasons.append(f"信号 {effect['signal']}：robust z={effect_z:.2f} ≥ {z_gate}，"
                           f"change_ratio={effect['change_ratio']:.3f} ≥ "
                           f"{1.0 + min_relative_effect:.2f}，显著且达最小效应。")
        else:
            reasons.extend(notes)

    return {"weight": _LEVELS[score], "weight_score": score, "evidence": reasons,
            "effect_z": effect_z, "effect_ratio": effect_ratio}


def _order_resource_ties(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within each weight tier, reorder resource candidates by change_ratio (desc).

    Only the slots already occupied by resource candidates (with a change_ratio)
    are permuted; every other candidate keeps its position. Python's sort is
    stable, so equal ratios keep their previous (Stage A / SHAP) order.
    """
    result = list(items)
    tiers: dict[int, list[int]] = {}
    for index, item in enumerate(result):
        if item.get("stage_b_change_ratio") is not None:
            tiers.setdefault(item["stage_b_weight_score"], []).append(index)
    for slots in tiers.values():
        ordered = sorted((result[i] for i in slots), key=lambda it: -it["stage_b_change_ratio"])
        for slot, item in zip(slots, ordered):
            result[slot] = item
    return result


def rerank_stage_b(stage_a: Mapping[str, Any], min_span_count: int = DEFAULT_MIN_SPAN_COUNT,
                   z_gate: float = Z_GATE,
                   min_relative_effect: float = MIN_RELATIVE_EFFECT,
                   order_ties_by_change: bool = ORDER_RESOURCE_TIES_BY_CHANGE) -> dict[str, Any]:
    """Return a Stage B result for a single Stage A result object."""
    result = dict(stage_a)
    stage_input = stage_a.get("stage_b_input") or {}
    metrics = [dict(item) for item in stage_a.get("ranked_metrics", [])]
    service = stage_input.get("service")
    service_key = _canonical_service(service)
    target_metrics = [item for item in metrics
                      if _canonical_service(item.get("service")) == service_key]

    if not stage_input.get("should_run") or not service or len(target_metrics) <= 1:
        result["stage_b"] = {
            "ran": False,
            "reason": "Stage A top service has at most one metric or should_run is false.",
            "min_span_count": min_span_count,
        }
        result["ranked_metrics"] = metrics
        return result

    evidence = stage_input.get("evidence") or {}
    trace = _trace_record(evidence, service)
    baseline_spans = int(_number(trace.get("baseline_spans")) or 0)
    anomaly_spans = int(_number(trace.get("anomaly_spans")) or 0)
    spans_sufficient = baseline_spans >= min_span_count and anomaly_spans >= min_span_count
    baseline_errors = int(_number(trace.get("baseline_error_spans")) or 0)
    anomaly_errors = int(_number(trace.get("anomaly_error_spans")) or 0)
    latency_ratio = _number(trace.get("p95_change_ratio"))
    has_error_spans = anomaly_errors > 0
    disk_anomaly, disk_signals = _filesystem_io_anomaly(evidence)

    annotated = []
    for stage_rank, item in enumerate(target_metrics):
        weight = _weight_for_metric(
            item, trace=trace, latency_ratio=latency_ratio,
            has_error_spans=has_error_spans, spans_sufficient=spans_sufficient,
            disk_anomaly=disk_anomaly, disk_signals=disk_signals, evidence=evidence,
            z_gate=z_gate, min_relative_effect=min_relative_effect,
        )
        annotated.append((stage_rank, {**item, "stage_b_weight": weight["weight"],
                                        "stage_b_weight_score": weight["weight_score"],
                                        "stage_b_evidence": weight["evidence"],
                                        "stage_b_effect_z": weight.get("effect_z"),
                                        "stage_b_change_ratio": weight.get("effect_ratio")}))

    # Sort: weight tier, then Stage A order, then SHAP (same as before).
    annotated.sort(key=lambda pair: (-pair[1]["stage_b_weight_score"],
                                     pair[0],
                                     -(_number(pair[1].get("shap")) or 0.0)))
    reranked_service_metrics = [item for _, item in annotated]
    if order_ties_by_change:
        reranked_service_metrics = _order_resource_ties(reranked_service_metrics)
    metric_iter = iter(reranked_service_metrics)
    reordered = [next(metric_iter) if _canonical_service(item.get("service")) == service_key else item
                 for item in metrics]
    for rank, item in enumerate(reordered, 1):
        item["rank"] = rank

    result["ranked_metrics"] = reordered
    result["stage_b"] = {
        "ran": True,
        "service": service,
        "min_span_count": min_span_count,
        "trace_evidence_sufficient": spans_sufficient,
        "trace_counts": {"baseline_spans": baseline_spans, "anomaly_spans": anomaly_spans,
                         "baseline_error_spans": baseline_errors,
                         "anomaly_error_spans": anomaly_errors},
        "latency_p95_change_ratio": latency_ratio,
        "loss_rule_triggered": _loss_triggered(spans_sufficient, has_error_spans, latency_ratio),
        "delay_rule_triggered": _delay_triggered(spans_sufficient, has_error_spans, latency_ratio),
        "disk_filesystem_io_anomaly": disk_anomaly,
        "disk_filesystem_io_signals": disk_signals,
        "resource_effect_available": bool((evidence.get("resource_effects") or {}).get("available")),
        "order_resource_ties_by_change": order_ties_by_change,
        "z_gate": z_gate,
        "min_relative_effect": min_relative_effect,
        "ranking": [item.get("name") for item in reranked_service_metrics],
    }
    return result


def main() -> int:
    """Run one Stage A JSON or every Stage A JSON in a folder."""
    project_root = Path(__file__).resolve().parents[2]
    # default_in = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageA_output"
    default_in = r"D:\projects\X-RAG\src\rag_llm\stageA_output_1014_ad_window"
    default_out = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageB_output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_in, help="Stage A JSON file or folder")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--z-gate", type=float, default=Z_GATE)
    parser.add_argument("--min-relative-effect", type=float, default=MIN_RELATIVE_EFFECT,
                        help="minimum fractional rise over baseline (0.10 = +10%%)")
    parser.add_argument("--tie-order", action="store_true",
                        help="ablation: order tied resource candidates by change_ratio")
    args = parser.parse_args()

    stage_a_input, output_dir = args.input, args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage_a_input.is_file():
        input_paths = [stage_a_input]
    elif stage_a_input.is_dir():
        input_paths = sorted(stage_a_input.glob("*_stage_a.json"))
    else:
        raise FileNotFoundError(f"Stage A 输入路径不存在：{stage_a_input}")
    if not input_paths:
        raise FileNotFoundError(f"没有找到 Stage A JSON：{stage_a_input}")

    failures = []
    for index, input_path in enumerate(input_paths, 1):
        try:
            stage_a = json.loads(input_path.read_text(encoding="utf-8"))
            result = rerank_stage_b(stage_a, z_gate=args.z_gate,
                                    min_relative_effect=args.min_relative_effect,
                                    order_ties_by_change=args.tie_order)
            output_path = output_dir / input_path.name.replace("_stage_a.json", "_stage_b.json")
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            stage_b = result.get("stage_b", {})
            top = (result.get("ranked_metrics") or [{}])[0].get("name")
            print(f"[{index}/{len(input_paths)}] {input_path.name}: ran={stage_b.get('ran')}, top_metric={top}")
        except Exception as exc:
            failures.append({"file": input_path.name, "error": str(exc)})
            print(f"[ERROR] {input_path.name}: {exc}")

    if failures:
        (output_dir / "stage_b_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
