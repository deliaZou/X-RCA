"""Generate Layer 2 candidates from PCA detections and KernelSHAP.

Ground-truth labels are retained only in the evaluation metadata; they do not
select SHAP points, define evidence windows, or affect candidate statistics.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import shap
from pyod.models.pca import PCA as PyodPCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================ CONFIG ============================

BASE = r"D:\projects\X-RCA\Dataset\RCAEval_ds\RE2-OB_preprocessed"
CONFIG = r"D:\projects\X-RCA\src\Layer1\selection.yaml"  # optional
# Keep the experiment output separate from existing Layer 2 results.
OUT = r"D:\projects\X-RCA\Dataset\RCAEval_ds\RE2-OB_layer1_output"

DEFAULT = {
    "pca_n_components": 0.95,
    "random_state": 42,
    "eval_quantile": 0.999,
    "kernel_background": 50,
    "max_explain": 15,
    "fidelity_topk": 5,
    # Merge detections separated by at most this many samples.
    # 0 means only directly adjacent detections are grouped.
    "ad_merge_gap_samples": 0,
    "ad_min_window_samples": 1,
}

# ================================================================


def load_config():
    cfg = dict(DEFAULT)
    if os.path.exists(CONFIG):
        try:
            import yaml

            with open(CONFIG, encoding="utf-8") as stream:
                config_data = yaml.safe_load(stream) or {}
            l1 = config_data.get("layer1", {}) or {}
            l2 = config_data.get("layer2", {}) or {}
            for key in ("pca_n_components", "random_state", "eval_quantile"):
                if key in l1:
                    cfg[key] = l1[key]
            for key in (
                "kernel_background", "max_explain", "fidelity_topk",
                "ad_merge_gap_samples", "ad_min_window_samples",
            ):
                if key in l2:
                    cfg[key] = l2[key]
            print(f"已读配置 {CONFIG}")
        except Exception as exc:
            print(f"读配置失败, 用内置默认: {exc}")
    else:
        print("未找到配置文件, 用内置默认")
    return cfg


def split_service(col):
    return col.rsplit("_", 1)[0] if "_" in col else col


def find_detection_windows(scores, threshold, merge_gap=0, min_length=1):
    """Convert thresholded point detections into inclusive [start, end] runs."""
    indices = np.flatnonzero(np.asarray(scores) > threshold)
    if len(indices) == 0:
        return []

    # A difference of gap + 1 means there are at most `gap` unflagged samples
    # between detections, so they belong to the same event window.
    split_at = np.where(np.diff(indices) > int(merge_gap) + 1)[0] + 1
    groups = np.split(indices, split_at)
    windows = [
        (int(group[0]), int(group[-1]))
        for group in groups
        if len(group) >= int(min_length)
    ]
    return windows


def _timestamp(test_time, index):
    if test_time is None:
        return None
    value = test_time[index]
    return value.item() if hasattr(value, "item") else value


def _empty_case_result(cid, row, meta, windows):
    gt_metric = meta["gt_metric"]
    return {
        "_meta": {
            "case_id": cid,
            "fold": int(row.fold),
            "ground_truth": {
                "service": meta["root_cause_service"],
                "fault": meta["fault_type"],
                "metric": gt_metric,
            },
        },
        "ad_windows": windows,
        "selected_ad_window": None,
        "n_explained_points": 0,
        "attribution_fidelity": None,
        "service_list": [],
        "metrics_list": [],
    }


def process_case(cid, row, features, cfg):
    cdir = os.path.join(BASE, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    # Labels are deliberately not loaded or used to generate candidates.
    with open(os.path.join(cdir, "meta.json"), encoding="utf-8") as stream:
        meta = json.load(stream)
    time_path = os.path.join(cdir, "test_time.npy")
    test_time = np.load(time_path).tolist() if os.path.exists(time_path) else None

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)

    detector = PyodPCA(
        n_components=cfg["pca_n_components"],
        random_state=cfg["random_state"],
    ).fit(Ztr)

    def score(X):
        return detector.decision_function(X)  # larger means more anomalous

    train_scores = score(Ztr)
    test_scores = score(Zte)
    threshold = np.quantile(train_scores, cfg["eval_quantile"])

    windows = find_detection_windows(
        test_scores,
        threshold,
        merge_gap=cfg["ad_merge_gap_samples"],
        min_length=cfg["ad_min_window_samples"],
    )
    window_json = [
        {
            "start_index": start,
            "end_index": end,
            "start_timestamp": _timestamp(test_time, start),
            "end_timestamp": _timestamp(test_time, end),
            "duration_samples": end - start + 1,
            "peak_score": float(np.max(test_scores[start:end + 1])),
        }
        for start, end in windows
    ]

    if not windows:
        result = _empty_case_result(cid, row, meta, window_json)
        summary = dict(
            case_id=cid, fault=row.fault, fidelity=None, n_points=0,
            ad_window_count=0,
            gt_in_top5_service=False,
        )
        return result, summary

    # The legacy candidate schema carries one interval per metric. Select the
    # AD event with the greatest cumulative score excess for that schema,
    # while retaining every detected window in `ad_windows` for auditability.
    start, end = max(
        windows,
        key=lambda w: float(np.maximum(test_scores[w[0]:w[1] + 1] - threshold, 0).sum()),
    )
    detected_points = np.flatnonzero(test_scores[start:end + 1] > threshold) + start
    sel = detected_points[np.argsort(-test_scores[detected_points])[:cfg["max_explain"]]]
    Zsel = Zte[sel]
    anomaly_indices = np.arange(start, end + 1, dtype=int)

    background = shap.sample(
        Ztr,
        min(cfg["kernel_background"], len(Ztr)),
        random_state=cfg["random_state"],
    )
    shap_values = np.asarray(
        shap.KernelExplainer(score, background).shap_values(Zsel, silent=True)
    )
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]
    feature_scores = np.abs(shap_values).mean(axis=0)

    def candidate_evidence(i):
        mu = Xtr[:, i].mean()
        sd = Xtr[:, i].std()
        anomaly_values = Xte[anomaly_indices, i]

        if sd < 1e-9:
            mean_delta = float(anomaly_values.mean() - mu) if len(anomaly_values) else 0.0
            return {
                "zscore_valid": False,
                "mean_zscore": None,
                "mean_abs_zscore": None,
                "max_abs_zscore": None,
                "direction": "up" if mean_delta > 0 else "down" if mean_delta < 0 else "stable",
                "duration_samples": 0,
                "anomaly_start_index": None,
                "anomaly_end_index": None,
                "anomaly_start_timestamp": None,
                "anomaly_end_timestamp": None,
                "trend": "unknown",
            }

        z = (Xte[:, i] - mu) / sd
        window_z = z[anomaly_indices]
        # Metric-specific |z| runs are searched only inside the AD window.
        active = (np.abs(z) >= 2) & (np.arange(len(z)) >= start) & (np.arange(len(z)) <= end)
        active_indices = np.flatnonzero(active)
        if len(active_indices):
            split_at = np.where(np.diff(active_indices) > 1)[0] + 1
            runs = np.split(active_indices, split_at)
            best_run = max(runs, key=len)
            best_start, best_end = int(best_run[0]), int(best_run[-1])
            duration = len(best_run)
        else:
            best_start = best_end = None
            duration = 0

        if len(anomaly_indices) >= 2:
            slope = np.polyfit(anomaly_indices, anomaly_values, 1)[0]
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        else:
            trend = "unknown"

        mean_z = float(window_z.mean()) if len(window_z) else 0.0
        return {
            "zscore_valid": True,
            "mean_zscore": round(mean_z, 5),
            "mean_abs_zscore": round(float(np.abs(window_z).mean()), 5) if len(window_z) else 0.0,
            "max_abs_zscore": round(float(np.abs(window_z).max()), 5) if len(window_z) else 0.0,
            "direction": "up" if mean_z > 0 else "down" if mean_z < 0 else "stable",
            "duration_samples": int(duration),
            "anomaly_start_index": best_start,
            "anomaly_end_index": best_end,
            "anomaly_start_timestamp": _timestamp(test_time, best_start) if best_start is not None else None,
            "anomaly_end_timestamp": _timestamp(test_time, best_end) if best_end is not None else None,
            "trend": trend,
        }

    allowed_metrics = {"cpu", "latency", "diskio", "mem", "socket"}
    eligible_indices = [
        i for i, col in enumerate(features)
        if col.rsplit("_", 1)[-1] in allowed_metrics
    ]

    service_scores = {}
    for i in eligible_indices:
        service = split_service(features[i])
        service_scores[service] = service_scores.get(service, 0.0) + float(feature_scores[i])
    service_rank = sorted(service_scores, key=service_scores.get, reverse=True)
    service_list = [
        {"service": service, "service_shap": round(service_scores[service], 5)}
        for service in service_rank
    ]

    metrics_list = []
    for i in eligible_indices:
        evidence = candidate_evidence(i)
        metrics_list.append({
            "name": features[i],
            "shap": round(float(feature_scores[i]), 5),
            **evidence,
        })
    metrics_list.sort(key=lambda metric: metric["shap"], reverse=True)

    topk = np.argsort(-feature_scores)[:cfg["fidelity_topk"]]
    masked = Zsel.copy()
    masked[:, topk] = 0.0
    full_score = score(Zsel).mean()
    masked_score = score(masked).mean()
    baseline_score = score(np.zeros((1, Zsel.shape[1]))).item()
    denom = abs(full_score - baseline_score) + 1e-9
    fidelity = float(max(0.0, (full_score - masked_score) / denom))

    result = {
        "_meta": {
            "case_id": cid,
            "fold": int(row.fold),
            "ground_truth": {
                "service": meta["root_cause_service"],
                "fault": meta["fault_type"],
                "metric": meta["gt_metric"],
            },
        },
        "ad_threshold": float(threshold),
        "ad_windows": window_json,
        "selected_ad_window": window_json[windows.index((start, end))],
        "n_explained_points": int(len(sel)),
        "attribution_fidelity": round(fidelity, 4),
        "service_list": service_list,
        "metrics_list": metrics_list,
    }
    summary = dict(
        case_id=cid,
        fault=row.fault,
        fidelity=round(fidelity, 4),
        n_points=int(len(sel)),
        ad_window_count=len(windows),
        gt_in_top5_service=meta["root_cause_service"] in service_rank[:5],
    )
    return result, summary


def run():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    print("配置:", cfg)

    manifest = pd.read_csv(os.path.join(BASE, "manifest.csv"))
    manifest = manifest[manifest.status == "ok"].reset_index(drop=True)
    with open(os.path.join(BASE, "feature_names.json"), encoding="utf-8") as stream:
        features = json.load(stream)

    summaries = []
    for i, row in manifest.iterrows():
        result, summary = process_case(row.case_id, row, features, cfg)
        output_path = os.path.join(OUT, f"{row.case_id}.json")
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=1)
        summaries.append(summary)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(manifest)}")

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(os.path.join(OUT, "_summary.csv"), index=False)
    print(f"\n完成, 输出 {len(summary_frame)} 份 JSON 到 {OUT}")
    detected = summary_frame[summary_frame["ad_window_count"] > 0]
    print(f"检测到 AD 窗口的案例: {len(detected)}/{len(summary_frame)}")
    if not detected.empty:
        print("案例级 fidelity 分布, 按故障类型:")
        print(detected.groupby("fault")["fidelity"].agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    run()
