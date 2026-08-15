from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.result_analysis import diagnose_public_summary


def render_report(diagnostics):
    zero = diagnostics[diagnostics["f1_sd_exact_zero"]]
    collapsed = diagnostics[diagnostics["degenerate_stability"]]
    eligible = diagnostics[~diagnostics["degenerate_stability"]]
    stability_winners = eligible.loc[eligible.groupby(["task", "model"])["f1_sd"].idxmin()]
    stability_counts = stability_winners["method"].value_counts()
    lines = [
        "# Seed Variance and Prediction-Collapse Audit",
        "",
        "## 결론",
        "",
        f"- 공개 요약표 {len(diagnostics)}개 조건 중 `f1_sd=0`은 {len(zero)}개다.",
        f"- 그중 단일-class 예측의 metric fingerprint를 정확히 만족하는 조건은 {len(collapsed)}개다.",
        "- CSV에는 원래 부동소수점 정밀도가 저장되어 있으므로 표시 반올림으로 0이 된 것이 아니다.",
        "- 이 네 행은 단일-class prediction과 일치하는 aggregate fingerprint이며, 원본 prediction 부재 때문에 과거 run의 collapse를 직접 확인한 것은 아니다.",
        "",
        "## 붕괴 조건",
        "",
        "| Task | Model | Method | Seeds | Macro-F1 | F1 SD | Classes | 판정 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in collapsed.to_dict("records"):
        lines.append(
            f"| {row['task']} | {row['model']} | {row['method']} | {row['seeds']} | "
            f"{row['f1_mean']:.12f} | {row['f1_sd']:.12f} | {row['num_labels']} | fingerprint-consistent with constant-class collapse |"
        )
    lines.extend([
        "",
        "## 판정 원리",
        "",
        "K-class 분류기가 모든 표본을 하나의 class로 예측하고 그 class의 실제 비율을 p라고 하면 "
        "`accuracy=p`, `macro_precision=p/K`, `macro_recall=1/K`, "
        "`macro_f1=(2p/(1+p))/K`가 된다. 위 조건들은 저장된 네 metric이 이 관계를 "
        "1e-12 절대오차 이내에서 동시에 만족한다.",
        "",
        "## 후속 실험",
        "",
        "원래 2-epoch 조건을 기준선으로 유지하고, 학습 epoch 증가, class-weighted loss, 상향 learning rate를 "
        "분리한 뒤 결합 조건까지 비교한다. 각 run에는 예측 class 수, 최대 class 예측률, "
        "예측 entropy, label count를 저장해 같은 문제가 다시 숨지 않도록 한다.",
        "",
        "## 붕괴 행 제외 stability sensitivity analysis",
        "",
        "| Method | Lowest-SD conditions after exclusion |",
        "|---|---:|",
    ])
    for method, count in stability_counts.items():
        lines.append(f"| {method} | {count} |")
    lines.extend([
        "",
        "이 순위도 원본의 나머지 run-level prediction을 직접 검사한 결과는 아니므로 sensitivity analysis로만 해석한다.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "final_tables" / "summary_by_task_model_method.csv")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "results_summary" / "variance_diagnostics.csv")
    parser.add_argument("--report-output", type=Path, default=ROOT / "results_summary" / "variance_diagnostics.md")
    args = parser.parse_args()

    diagnostics = diagnose_public_summary(args.input)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.csv_output, index=False, encoding="utf-8-sig", float_format="%.17g")
    args.report_output.write_text(render_report(diagnostics), encoding="utf-8")
    collapsed = diagnostics[diagnostics["degenerate_stability"]]
    print(f"rows={len(diagnostics)} zero_sd={int(diagnostics['f1_sd_exact_zero'].sum())} collapsed={len(collapsed)}")
    for row in collapsed.itertuples():
        print(row.task, row.model, row.method, f"f1={row.f1_mean:.12f}", f"sd={row.f1_sd:.12f}")


if __name__ == "__main__":
    main()
