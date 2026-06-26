from __future__ import annotations

import datetime
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ["full_ft", "lora", "adapter", "ia3", "bitfit"]
METHOD_LABEL = {
    "full_ft": "Full Fine-tuning",
    "lora": "LoRA",
    "adapter": "Adapter",
    "ia3": "IA³",
    "bitfit": "BitFit",
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value, digits=6) -> str:
    return f"{value:.{digits}f}"


def pct(value) -> str:
    return f"{value * 100:.2f}%"


def minutes(seconds) -> str:
    return f"{seconds / 60:.1f}분"


def load_rows():
    latest = {}
    for path in RESULTS.rglob("final_metrics.json"):
        payload = read_json(path)
        if not payload:
            continue
        key = (
            payload.get("study"),
            payload.get("task"),
            payload.get("model"),
            payload.get("method"),
            payload.get("seed"),
        )
        mtime = path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, path, payload)

    rows = []
    for _, _, item in latest.values():
        rows.append(
            {
                "study": item.get("study", ""),
                "task": item.get("task", ""),
                "model": item.get("model", ""),
                "method": item.get("method", ""),
                "seed": item.get("seed", ""),
                "f1": float(item.get("test_macro_f1", 0) or 0),
                "accuracy": float(item.get("test_accuracy", 0) or 0),
                "precision": float(item.get("test_macro_precision", 0) or 0),
                "recall": float(item.get("test_macro_recall", 0) or 0),
                "seconds": float(item.get("train_seconds", 0) or 0),
                "train_rows": int(item.get("train_rows", 0) or 0),
                "val_rows": int(item.get("validation_rows", 0) or 0),
                "test_rows": int(item.get("test_rows", 0) or 0),
                "epochs": int(item.get("epochs_requested", 0) or 0),
                "params": int(item.get("trainable_params", 0) or 0),
                "ratio": float(item.get("trainable_parameter_ratio", 0) or 0),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["task"], row["model"], row["method"])].append(row)

    summaries = []
    for (study, task, model, method), group in grouped.items():
        first = group[0]
        summaries.append(
            {
                "study": study,
                "task": task,
                "model": model,
                "method": method,
                "n": len(group),
                "f1_mean": mean([x["f1"] for x in group]),
                "f1_sd": sd([x["f1"] for x in group]),
                "accuracy": mean([x["accuracy"] for x in group]),
                "precision": mean([x["precision"] for x in group]),
                "recall": mean([x["recall"] for x in group]),
                "seconds": mean([x["seconds"] for x in group]),
                "params": round(mean([x["params"] for x in group])),
                "ratio": mean([x["ratio"] for x in group]),
                "train_rows": first["train_rows"],
                "val_rows": first["val_rows"],
                "test_rows": first["test_rows"],
                "epochs": first["epochs"],
            }
        )

    full = {
        (x["study"], x["task"], x["model"]): x
        for x in summaries
        if x["method"] == "full_ft"
    }
    for item in summaries:
        base = full.get((item["study"], item["task"], item["model"]))
        if base:
            item["delta_full"] = item["f1_mean"] - base["f1_mean"]
            item["time_saving"] = (
                (base["seconds"] - item["seconds"]) / base["seconds"]
                if base["seconds"]
                else 0
            )
        else:
            item["delta_full"] = 0.0
            item["time_saving"] = 0.0
    return summaries


def table(headers, rows):
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def method_table(items):
    order = {method: idx for idx, method in enumerate(METHODS)}
    ranked = sorted(items, key=lambda x: (-x["f1_mean"], order.get(x["method"], 99)))
    rows = []
    for rank, item in enumerate(ranked, 1):
        rank_html = f'<span class="best">{rank}</span>' if rank == 1 else str(rank)
        sign = "+" if item["delta_full"] >= 0 else ""
        saving_class = "pos" if item["time_saving"] >= 0 else "neg"
        rows.append(
            [
                rank_html,
                esc(METHOD_LABEL.get(item["method"], item["method"])),
                str(item["n"]),
                f"{fmt(item['f1_mean'])} ± {fmt(item['f1_sd'])}",
                f"{sign}{fmt(item['delta_full'])}",
                f'<span class="{saving_class}">{item["time_saving"] * 100:.2f}%</span>',
                minutes(item["seconds"]),
                f"{item['params']:,}",
                pct(item["ratio"]),
            ]
        )
    return table(
        [
            "순위",
            "방법",
            "Seed 수",
            "Test Macro-F1 ± 표준편차",
            "Full FT 대비 F1",
            "Full FT 대비 시간 절감",
            "평균 학습시간",
            "학습 파라미터",
            "학습 파라미터 비율",
        ],
        rows,
    )


def progress():
    out = {}
    for study in ["study1", "study2", "study3"]:
        out[study] = read_json(RESULTS / study / "PAPER" / "progress.json") or {}
    return out


def generate():
    rows = load_rows()
    summary = summarize(rows)
    prog = progress()

    completed1 = int(prog.get("study1", {}).get("completed", 0) or 0)
    completed2 = int(prog.get("study2", {}).get("completed", 0) or 0)
    completed3 = int(prog.get("study3", {}).get("completed", 0) or 0)
    total_done = completed1 + completed2 + completed3
    remaining2 = max(0, 450 - completed2)
    current = prog.get("study2", {}).get("current", {}) or {}

    # 2026-06-25 정오 기준 남은 task 구성 기반 보정치.
    low_hours = remaining2 / 251 * 10.5 if remaining2 else 0
    high_hours = remaining2 / 251 * 12.0 if remaining2 else 0
    now = datetime.datetime.now()
    finish_low = now + datetime.timedelta(hours=low_hours)
    finish_high = now + datetime.timedelta(hours=high_hours)

    study2_counts = defaultdict(int)
    for row in rows:
        if row["study"] == "study2":
            study2_counts[(row["task"], row["model"])] += 1

    best_counts = defaultdict(int)
    for key in sorted(
        set((x["study"], x["task"], x["model"]) for x in summary if x["n"] >= 5)
    ):
        candidates = [
            x
            for x in summary
            if x["n"] >= 5 and (x["study"], x["task"], x["model"]) == key
        ]
        if len(candidates) >= 5:
            best = max(candidates, key=lambda x: x["f1_mean"])
            best_counts[METHOD_LABEL.get(best["method"], best["method"])] += 1
    best_text = ", ".join(f"{k} {v}회" for k, v in sorted(best_counts.items(), key=lambda x: -x[1]))

    def section_for(study: str):
        html_parts = []
        for task, model in sorted(
            set((x["task"], x["model"]) for x in summary if x["study"] == study)
        ):
            items = [
                x
                for x in summary
                if x["study"] == study and x["task"] == task and x["model"] == model
            ]
            first = items[0]
            html_parts.append(
                f"""
                <section class="card">
                  <h3>{esc(study.upper())} · {esc(task)} · {esc(model)}</h3>
                  <p class="meta">데이터: train {first['train_rows']:,} / validation {first['val_rows']:,} / test {first['test_rows']:,} · Epoch {first['epochs']}</p>
                  {method_table(items)}
                </section>
                """
            )
        return "\n".join(html_parts)

    study2_overview = table(
        ["Task", "Model", "완료 run", "예상 run"],
        [[esc(k[0]), esc(k[1]), str(v), "25"] for k, v in sorted(study2_counts.items())],
    )

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2457d6;--green:#11845b;--red:#b42318}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.55}
    .wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}.hero{background:linear-gradient(135deg,#172033,#2457d6);color:#fff;border-radius:24px;padding:34px;box-shadow:0 18px 45px rgba(25,40,80,.18)}
    h1{margin:0 0 10px;font-size:31px;letter-spacing:-.03em} h2{margin:34px 0 14px;font-size:22px;letter-spacing:-.02em} h3{margin:0 0 8px;font-size:17px}
    .hero p{margin:6px 0;color:#e9efff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:16px}.stat b{display:block;font-size:25px}.stat span{font-size:13px;color:#dbe5ff}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin:14px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}.meta{color:var(--muted);font-size:13px;margin:4px 0 14px}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px 16px;border-radius:12px;color:#2a3345}
    table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle} th{background:#f1f4fa;color:#465164;font-weight:700}
    .best{display:inline-block;background:#eaf1ff;color:var(--blue);font-weight:800;padding:2px 8px;border-radius:999px}.pos{color:var(--green);font-weight:700}.neg{color:var(--red);font-weight:700}.small{font-size:12px;color:#dbe5ff}
    ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}@media(max-width:900px){.grid,.cols{grid-template-columns:1fr}.wrap{padding:20px 12px}.hero{padding:24px}table{font-size:12px}}
    """

    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine-tuning Strategy Study 최신 진행 보고서</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>Fine-tuning Strategy Study 진행 보고서</h1>
    <p>5개 파인튜닝 방법을 3개 Study와 5개 seed 반복으로 비교하는 실험입니다. 성능, 안정성, 학습시간, 학습 파라미터 효율을 함께 평가합니다.</p>
    <p class="small">최신화 시각: {now.strftime('%Y-%m-%d %H:%M:%S')} KST · 실행 중인 학습은 건드리지 않고 결과 파일만 읽어 반영</p>
    <div class="grid">
      <div class="stat"><b>{total_done}/550</b><span>전체 완료 run</span></div>
      <div class="stat"><b>{completed2}/450</b><span>Study 2 완료 run</span></div>
      <div class="stat"><b>{remaining2}</b><span>Study 2 남은 run</span></div>
      <div class="stat"><b>{low_hours:.1f}~{high_hours:.1f}h</b><span>남은 task 구성 기준 추정</span></div>
    </div>
  </section>

  <h2>1. 현재 진행상황</h2>
  <section class="card">
    {table(["Study", "상태", "완료/전체", "비고"], [
      ["Study 1", "완료", f"{completed1}/25", "BERTweet + Measuring Hate Speech"],
      ["Study 3", "완료", f"{completed3}/75", "KLUE RoBERTa + 한국어 데이터셋 3개"],
      ["Study 2", "진행 중", f"{completed2}/450", f"현재: {esc(current.get('task_key',''))} / {esc(current.get('model_name',''))} / {esc(current.get('method',''))} / seed {esc(current.get('seed',''))}"],
    ])}
    <p class="meta">예상 종료 범위: {finish_low.strftime('%Y-%m-%d %H:%M')} ~ {finish_high.strftime('%Y-%m-%d %H:%M')} KST. 남은 news_topic 구간 속도에 따라 변동될 수 있습니다.</p>
  </section>

  <h2>2. 현재까지의 핵심 인사이트</h2>
  <section class="card">
    <div class="cols">
      <div class="note"><b>Full FT가 항상 압도적인 정답은 아닙니다.</b><br>Study 3의 YNAT에서는 Adapter가 Full FT보다 높은 평균 Macro-F1을 보였습니다.</div>
      <div class="note"><b>어려운 데이터에서는 Full FT가 여전히 강합니다.</b><br>NSMC와 K-MHaS에서는 Full FT가 가장 높은 성능을 보였습니다.</div>
      <div class="note"><b>LoRA는 파라미터 효율과 시간 효율이 다릅니다.</b><br>학습 파라미터는 크게 줄지만, 실제 wall-clock 시간은 Full FT보다 길거나 비슷한 경우가 있었습니다.</div>
      <div class="note"><b>현재까지 완료 그룹의 1위 분포</b><br>{esc(best_text)}. 최종적으로 도메인, 데이터 크기, 모델 적합성과 연결해 해석할 수 있습니다.</div>
    </div>
  </section>

  <h2>3. Study 1 확정 결과</h2>
  {section_for("study1")}

  <h2>4. Study 2 중간 결과</h2>
  <section class="card">
    <p>Study 2는 BERTweet과 범용 RoBERTa를 여러 영어 데이터셋에서 비교하는 핵심 확장 실험입니다. 현재 완료된 결과는 중간 해석에는 사용할 수 있지만, 최종 결론은 전체 450개 run 완료 후 확정하는 것이 안전합니다.</p>
    {study2_overview}
  </section>
  {section_for("study2")}

  <h2>5. Study 3 확정 결과</h2>
  {section_for("study3")}

  <h2>6. 현재 기준 종합 결론</h2>
  <section class="card">
    <ul>
      <li>단일 F1 점수만으로 최고의 파인튜닝 방법을 정하는 것은 부족합니다.</li>
      <li>Adapter는 성능, 시간, 파라미터 효율의 균형이 가장 좋은 후보로 보입니다.</li>
      <li>Full FT는 여전히 가장 강한 기준선이며, 어려운 데이터에서 성능 우위가 뚜렷합니다.</li>
      <li>LoRA는 parameter-efficient이지만, 반드시 time-efficient하지는 않습니다.</li>
      <li>BitFit과 IA³는 가볍지만 성능 손실이 커서 초경량 기준선으로 해석하는 것이 적절합니다.</li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    return doc


def main():
    doc = generate()
    outputs = [
        ROOT / "PROFESSOR_PROGRESS_REPORT_LATEST.html",
        ROOT / "PROFESSOR_PROGRESS_REPORT_2026-06-24.html",
        ROOT / "PROFESSOR_BRIEF_OVERVIEW_2026-06-24.html",
    ]
    for path in outputs:
        path.write_text(doc, encoding="utf-8-sig")
        print(path.resolve(), path.stat().st_size)


if __name__ == "__main__":
    main()
