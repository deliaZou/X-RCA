import csv
import json
from pathlib import Path


SUMMARY_METRICS = ("AC@1", "AC@3", "AC@5", "Avg@3", "Avg@5")


def _ranking_metrics(records, rank_key):
    """Return AC@1/3/5 and Avg@3/5 from the per-case root-cause ranks."""
    if not records:
        return [""] * len(SUMMARY_METRICS)

    ranks = [r[rank_key] for r in records]
    ac = {
        k: sum(rank != "" and rank <= k for rank in ranks) / len(ranks)
        for k in range(1, 6)
    }
    avg3 = sum(ac[k] for k in range(1, 4)) / 3
    avg5 = sum(ac[k] for k in range(1, 6)) / 5
    values = (ac[1], ac[3], ac[5], avg3, avg5)
    return [f"{value:.2%}" for value in values]


def _summary_row(label, records):
    return [label, *_ranking_metrics(records, "metric_rank"), "",
            label, *_ranking_metrics(records, "service_rank")]


def process_folder_to_csv(folder_path, output_csv):
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))

    if not json_files:
        print(f"未在路径 '{folder_path}' 中找到 JSON 文件。")
        return

    # 定义 CSV 表头（在 GT_Service 后加入 GT_Fault）
    headers = [
        "Case_ID",
        "GT_Service",
        "GT_Fault",
        "Service_Rank",
        "GT_Metric",
        "Metric_Rank",
        # Top 1-5 Services
        "Top1_Service",
        "Top1_Service_SHAP",
        "Top2_Service",
        "Top2_Service_SHAP",
        "Top3_Service",
        "Top3_Service_SHAP",
        "Top4_Service",
        "Top4_Service_SHAP",
        "Top5_Service",
        "Top5_Service_SHAP",
        # Top 1-5 Metrics
        "Top1_Metric",
        "Top1_Metric_SHAP",
        "Top1_Metric_Direction",
        "Top2_Metric",
        "Top2_Metric_SHAP",
        "Top2_Metric_Direction",
        "Top3_Metric",
        "Top3_Metric_SHAP",
        "Top3_Metric_Direction",
        "Top4_Metric",
        "Top4_Metric_SHAP",
        "Top4_Metric_Direction",
        "Top5_Metric",
        "Top5_Metric_SHAP",
        "Top5_Metric_Direction",
    ]

    rows = []
    summary_records = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("_meta", {})
            case_id = meta.get("case_id", file_path.stem)
            ground_truth = meta.get("ground_truth", {})
            gt_service = ground_truth.get("service", "")
            gt_fault = ground_truth.get("fault", "")
            gt_metric = ground_truth.get("metric", "")

            # 1. 计算 Service 排名
            service_list = data.get("service_list", [])
            service_rank = ""
            for idx, s in enumerate(service_list, start=1):
                if s.get("service") == gt_service:
                    service_rank = idx
                    break

            # 2. 计算 Metric 排名
            metrics_list = data.get("metrics_list", [])
            metric_rank = ""
            for idx, m in enumerate(metrics_list, start=1):
                if m.get("name") == gt_metric:
                    metric_rank = idx
                    break

            # 组装基础信息行
            row = [
                case_id,
                gt_service,
                gt_fault,
                service_rank,
                gt_metric,
                metric_rank,
            ]

            # 3. 补充 Top 1-5 Service 及其 SHAP 值
            for i in range(5):
                if i < len(service_list):
                    s = service_list[i]
                    row.extend(
                        [s.get("service", ""), s.get("service_shap", "")]
                    )
                else:
                    row.extend(["", ""])

            # 4. 补充 Top 1-5 Metric、SHAP 值及 Direction
            for i in range(5):
                if i < len(metrics_list):
                    m = metrics_list[i]
                    row.extend(
                        [
                            m.get("name", ""),
                            m.get("shap", ""),
                            m.get("direction", ""),
                        ]
                    )
                else:
                    row.extend(["", "", ""])

            rows.append(row)
            summary_records.append({
                "fault": gt_fault,
                "service_rank": service_rank,
                "metric_rank": metric_rank,
            })

        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")

    # 写入 CSV 文件
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"已成功提取 {len(rows)} 个案例，统计结果存至: {output_csv}")

    # Write fine- and coarse-grained summaries side by side.
    summary_path = Path(output_csv).with_name(f"{Path(output_csv).stem}_summary.csv")
    summary_headers = [
        "Fine_GT_Fault", "AC@1", "AC@3", "AC@5",
        "Avg@3", "Avg@5", "",
        "Coarse_GT_Fault", "AC@1", "AC@3", "AC@5",
        "Avg@3", "Avg@5",
    ]
    fault_values = sorted({r["fault"] for r in summary_records if r["fault"]})
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)
        for fault in fault_values:
            writer.writerow(_summary_row(fault, [r for r in summary_records if r["fault"] == fault]))
        writer.writerow(_summary_row("Overall", summary_records))
    print(f"Fine/Coarse 指标汇总已保存至: {summary_path}")


if __name__ == "__main__":
    TARGET_FOLDER = r"D:\projects\X-RCA\Dataset\RCAEval_ds\RE2-OB_layer1_output"
    OUTPUT_CSV = TARGET_FOLDER + r"\root_cause_analysis_results.csv"
    process_folder_to_csv(TARGET_FOLDER, OUTPUT_CSV)
