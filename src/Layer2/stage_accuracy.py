"""Evaluate Stage A and Stage B coarse/fine ranking accuracy."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


K_VALUES = (1, 3, 5)
MAX_K = max(K_VALUES)
METRIC_KEYS = tuple(
    [f"AC@{k}_{kind}" for kind in ("coarse", "fine") for k in K_VALUES]
    + [f"Avg@{k}_{kind}" for kind in ("coarse", "fine") for k in (3, 5)]
)


def _canonical_service(value: Any) -> str:
    return re.sub(r"service$", "", str(value or "").strip().lower())


def _load_json_files(directory: Path, pattern: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    result = {}
    for path in sorted(directory.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = payload.get("_meta") or {}
        case_id = str(payload.get("case_id") or meta.get("case_id") or path.stem)
        result[case_id] = (path, payload)
    return result


def _ground_truth(l2: Mapping[str, Any], case_id: str) -> dict[str, str] | None:
    meta = l2.get("_meta") or {}
    gt = meta.get("ground_truth") or l2.get("ground_truth") or {}
    service = gt.get("service")
    metric = gt.get("metric")
    if not service or not metric:
        return None
    parts = case_id.rsplit("_", 2)
    fault_type = str(gt.get("fault") or (parts[-2] if len(parts) == 3 else "unknown")).lower()
    return {"service": str(service), "metric": str(metric), "fault_type": fault_type}


def _ranking(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("ranked_metrics") or []
    return [row for row in rows if isinstance(row, Mapping)]


def _case_scores(ranking: list[dict[str, Any]], gt_service: str, gt_metric: str) -> dict[str, float]:
    # Preserve the supplied rank order; Stage A/B output is already ordered.
    coarse_hit, fine_hit = [], []
    for row in ranking[:MAX_K]:
        coarse_hit.append(_canonical_service(row.get("service")) == _canonical_service(gt_service))
        fine_hit.append(str(row.get("name") or "").strip().lower() == gt_metric.strip().lower())
    coarse_hit.extend([False] * (MAX_K - len(coarse_hit)))
    fine_hit.extend([False] * (MAX_K - len(fine_hit)))

    def cumulative(hits: list[bool], k: int) -> float:
        # Per-case accuracy is represented as a percentage (0 or 100).
        return 100.0 if any(hits[:k]) else 0.0

    values: dict[str, float] = {}
    for k in K_VALUES:
        values[f"AC@{k}_coarse"] = cumulative(coarse_hit, k)
        values[f"AC@{k}_fine"] = cumulative(fine_hit, k)
    for k in (3, 5):
        values[f"Avg@{k}_coarse"] = round(
            sum(cumulative(coarse_hit, j) for j in range(1, k + 1)) / k, 2
        )
        values[f"Avg@{k}_fine"] = round(
            sum(cumulative(fine_hit, j) for j in range(1, k + 1)) / k, 2
        )
    return values


def _mean_rows(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    return {key: (round(sum(float(row[key]) for row in rows) / len(rows), 2) if rows else None)
            for key in METRIC_KEYS}


def evaluate(stage_a_dir: Path, stage_b_dir: Path, l2_dir: Path) -> dict[str, Any]:
    stage_a_files = _load_json_files(stage_a_dir, "*_stage_a.json")
    stage_b_files = _load_json_files(stage_b_dir, "*_stage_b.json")
    l2_files = _load_json_files(l2_dir, "*.json")

    common_ids = sorted(set(stage_a_files) & set(stage_b_files) & set(l2_files))
    per_stage: dict[str, list[dict[str, Any]]] = {"stage_a": [], "stage_b": []}
    excluded = []
    for case_id in common_ids:
        l2 = l2_files[case_id][1]
        gt = _ground_truth(l2, case_id)
        if gt is None:
            excluded.append({"case_id": case_id, "reason": "missing ground_truth.service or ground_truth.metric"})
            continue
        for stage, files in (("stage_a", stage_a_files), ("stage_b", stage_b_files)):
            payload = files[case_id][1]
            scores = _case_scores(_ranking(payload), gt["service"], gt["metric"])
            per_stage[stage].append({"case_id": case_id, "fault_type": gt["fault_type"],
                                     "ground_truth_service": gt["service"],
                                     "ground_truth_metric": gt["metric"],
                                     "top1_metric": (_ranking(payload)[0].get("name") if _ranking(payload) else None),
                                     **scores})

    reports = {}
    for stage, cases in per_stage.items():
        by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            by_fault[case["fault_type"]].append(case)
        per_fault = {fault: {"n": len(rows), **_mean_rows(rows)}
                     for fault, rows in sorted(by_fault.items())}
        macro = {key: (round(sum(row[key] for row in per_fault.values()) / len(per_fault), 2)
                       if per_fault else None) for key in METRIC_KEYS}
        reports[stage] = {
            "evaluated_cases": len(cases),
            "per_fault_type": per_fault,
            "overall_micro": {"n": len(cases), **_mean_rows(cases)},
            "overall_macro_by_fault_type": {"n_fault_types": len(per_fault), **macro},
            "per_case": cases,
        }

    stage_a_ids, stage_b_ids = set(stage_a_files), set(stage_b_files)
    l2_ids = set(l2_files)
    return {
        "definition": {
            "coarse": "A hit occurs if any top-k metric belongs to the ground-truth service.",
            "fine": "A hit occurs if any top-k metric name exactly matches ground_truth.metric (full service_metric name).",
            "Avg@k": "For each case, Avg@k = (AC@1 + ... + AC@k) / k; values and reported means are percentages.",
            "overall_micro": "Pool all evaluated cases before averaging.",
            "overall_macro_by_fault_type": "Average the per-fault-type means equally.",
        },
        "input_counts": {"stage_a_files": len(stage_a_files), "stage_b_files": len(stage_b_files),
                         "l2_files": len(l2_files), "common_cases": len(common_ids)},
        "missing_case_ids": {
            "stage_a": sorted(l2_ids - stage_a_ids),
            "stage_b": sorted(l2_ids - stage_b_ids),
            "l2": sorted((stage_a_ids & stage_b_ids) - l2_ids),
        },
        "excluded": excluded,
        "stages": reports,
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    columns = ["stage", "fault_type", "n", "n_fault_types", *METRIC_KEYS]
    rows = []
    for stage, stage_report in report["stages"].items():
        for fault, metrics in stage_report["per_fault_type"].items():
            rows.append({"stage": stage, "fault_type": fault, **metrics})
        for name in ("overall_micro", "overall_macro_by_fault_type"):
            rows.append({"stage": stage,
                         "fault_type": "总计_micro" if name == "overall_micro" else "总计_macro",
                         **stage_report[name]})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    stage_a_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageA_output"
    stage_b_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageB_output"
    l2_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer1_output"
    output_dir = project_root / "Dataset" / "RCAEval_ds" / "RE2-OB_layer2_stageB_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate(stage_a_dir, stage_b_dir, l2_dir)
    json_path = output_dir / "stage_a_b_accuracy_summary.json"
    csv_path = output_dir / "stage_a_b_accuracy_by_fault.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, report)

    print(f"Matched L2/Stage A/Stage B cases: {report['input_counts']['common_cases']}")
    for stage, values in report["stages"].items():
        print(f"\n{stage.upper()} (n={values['overall_micro']['n']})")
        for fault, metrics in values["per_fault_type"].items():
            print(f"  {fault:>8} n={metrics['n']:>2}  "
                  f"coarse AC@1/3/5={metrics['AC@1_coarse']:.2f}%/"
                  f"{metrics['AC@3_coarse']:.2f}%/{metrics['AC@5_coarse']:.2f}%  "
                  f"fine AC@1/3/5={metrics['AC@1_fine']:.2f}%/"
                  f"{metrics['AC@3_fine']:.2f}%/{metrics['AC@5_fine']:.2f}%")
        for label, key in (("总计 micro", "overall_micro"), ("总计 macro", "overall_macro_by_fault_type")):
            metrics = values[key]
            print(f"  {label}  coarse AC@1/3/5={metrics['AC@1_coarse']:.2f}%/"
                  f"{metrics['AC@3_coarse']:.2f}%/{metrics['AC@5_coarse']:.2f}%  "
                  f"fine AC@1/3/5={metrics['AC@1_fine']:.2f}%/"
                  f"{metrics['AC@3_fine']:.2f}%/{metrics['AC@5_fine']:.2f}%")
    print(f"\nDetailed JSON: {json_path}")
    print(f"Fault-type summary CSV: {csv_path}")
    print(f"Missing Stage A cases: {len(report['missing_case_ids']['stage_a'])}; "
          f"missing Stage B cases: {len(report['missing_case_ids']['stage_b'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
