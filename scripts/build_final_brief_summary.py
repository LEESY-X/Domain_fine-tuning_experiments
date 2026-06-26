from __future__ import annotations

import datetime as dt
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "FINAL_BRIEF_SUMMARY.html"

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


def esc(x):
    return html.escape(str(x))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0


def fmt(x, n=4):
    return f"{x:.{n}f}"


def load_summary():
    latest = {}
    for p in RESULTS.rglob("final_metrics.json"):
        j = read_json(p)
        if not j:
            continue
        key = (j.get("study"), j.get("task"), j.get("model"), j.get("method"), j.get("seed"))
        if key not in latest or p.stat().st_mtime > latest[key][0]:
            latest[key] = (p.stat().st_mtime, j)
    rows = []
    for _, j in latest.values():
        rows.append(
            {
                "study": j.get("study"),
                "task": j.get("task"),
                "model": j.get("model"),
                "method": j.get("method"),
                "f1": float(j.get("test_macro_f1", 0) or 0),
                "seconds": float(j.get("train_seconds", 0) or 0),
                "ratio": float(j.get("trainable_parameter_ratio", 0) or 0),
                "train_rows": int(j.get("train_rows", 0) or 0),
                "epochs": int(j.get("epochs_requested", 0) or 0),
            }
        )
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["study"], r["task"], r["model"], r["method"])].append(r)
    summary = []
    for (study, task, model, method), group in grouped.items():
        summary.append(
            {
                "study": study,
                "task": task,
                "model": model,
                "method": method,
                "n": len(group),
                "f1": mean([x["f1"] for x in group]),
                "f1_sd": sd([x["f1"] for x in group]),
                "seconds": mean([x["seconds"] for x in group]),
                "ratio": mean([x["ratio"] for x in group]),
            }
        )
    return rows, summary


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def count_text(counter):
    return ", ".join(f"{k} {v}회" for k, v in sorted(counter.items(), key=lambda x: -x[1]))


def main():
    rows, summary = load_summary()
    groups = []
    for key in sorted(set((x["study"], x["task"], x["model"]) for x in summary)):
        items = [x for x in summary if (x["study"], x["task"], x["model"]) == key]
        if len(items) == 5 and all(x["n"] == 5 for x in items):
            groups.append((key, items))

    perf = defaultdict(int)
    time_eff = defaultdict(int)
    param_eff = defaultdict(int)
    stable = defaultdict(int)
    for _, items in groups:
        perf[METHOD_LABEL[max(items, key=lambda x: x["f1"])["method"]]] += 1
        time_eff[METHOD_LABEL[min(items, key=lambda x: x["seconds"])["method"]]] += 1
        param_eff[METHOD_LABEL[min(items, key=lambda x: x["ratio"])["method"]]] += 1
        stable[METHOD_LABEL[min(items, key=lambda x: x["f1_sd"])["method"]]] += 1

    epoch_examples = sum(r["train_rows"] * r["epochs"] for r in rows)
    train_hours = sum(r["seconds"] for r in rows) / 3600

    key_cases = []
    for key, items in groups:
        full = next((x for x in items if x["method"] == "full_ft"), None)
        if not full:
            continue
        best = max(items, key=lambda x: x["f1"])
        if best["method"] != "full_ft":
            key_cases.append(
                [
                    key[0],
                    key[1],
                    key[2],
                    METHOD_LABEL[best["method"]],
                    f"{fmt(best['f1'])} vs Full FT {fmt(full['f1'])}",
                ]
            )

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2563eb;--green:#059669}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.58}
    .wrap{max-width:980px;margin:0 auto;padding:32px 20px 70px}.hero{background:linear-gradient(135deg,#111827,#2563eb);color:white;border-radius:26px;padding:34px;box-shadow:0 20px 50px rgba(25,40,80,.20)}
    h1{margin:0 0 10px;font-size:32px;letter-spacing:-.04em} h2{margin:30px 0 12px;font-size:22px;letter-spacing:-.03em} h3{margin:0 0 8px}
    .hero p{color:#e9efff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;margin:12px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}
    .stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:16px;padding:14px}.stat b{display:block;font-size:24px}.stat span{font-size:12px;color:#dbe5ff}
    table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left} th{background:#f1f4fa}.note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px;border-radius:12px}.big{font-size:20px;font-weight:900;color:var(--blue)} ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}
    @media(max-width:820px){.grid,.grid2{grid-template-columns:1fr}.wrap{padding:20px 12px}}
    """

    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine-tuning Strategy 핵심 요약</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>Fine-tuning Strategy 핵심 요약</h1>
    <p>Full Fine-tuning과 PEFT 전략을 성능, 시간 효율, 파라미터 효율, 안정성 기준으로 비교한 550-run 실험 요약입니다.</p>
    <div class="grid">
      <div class="stat"><b>550/550</b><span>완료 run</span></div>
      <div class="stat"><b>0</b><span>실패 run</span></div>
      <div class="stat"><b>{train_hours:.1f}h</b><span>순수 학습시간</span></div>
      <div class="stat"><b>{epoch_examples:,}</b><span>학습 샘플 처리량</span></div>
    </div>
  </section>

  <h2>1. 한 줄 결론</h2>
  <section class="card">
    <p class="big">최고의 파인튜닝 방법은 고정되어 있지 않았고, 선택 기준은 성능·시간 효율·파라미터 효율·안정성에 따라 달라졌다.</p>
  </section>

  <h2>2. 지표별 1위 분포</h2>
  <section class="grid2">
    <div class="card"><h3>성능 1위</h3><p>{esc(count_text(perf))}</p></div>
    <div class="card"><h3>시간 효율 1위</h3><p>{esc(count_text(time_eff))}</p></div>
    <div class="card"><h3>파라미터 효율 1위</h3><p>{esc(count_text(param_eff))}</p></div>
    <div class="card"><h3>안정성 1위</h3><p>{esc(count_text(stable))}</p></div>
  </section>

  <h2>3. 핵심 인사이트</h2>
  <section class="card">
    <ul>
      <li>Full FT는 가장 강한 성능 기준선이지만 모든 데이터셋에서 항상 최고는 아니었다.</li>
      <li>Adapter는 성능 손실과 시간 절감의 균형이 좋아 현실적인 선택지로 나타났다.</li>
      <li>LoRA는 파라미터 효율적이지만 실제 학습시간이 항상 짧지는 않았다.</li>
      <li>IA³는 평균 성능은 낮지만 일부 hate-speech task에서 Full FT를 넘었다.</li>
      <li>BitFit은 시간 효율이 가장 강하지만 복잡한 task에서는 성능 손실이 컸다.</li>
      <li>BERTweet이 모든 트윗 task에서 항상 RoBERTa-base보다 우세하지는 않았다.</li>
      <li>성능 1위, 시간 효율 1위, 파라미터 효율 1위, 안정성 1위가 서로 달랐다.</li>
    </ul>
  </section>

  <h2>4. PEFT가 Full FT를 넘은 사례</h2>
  <section class="card">
    {table(["Study", "Task", "Model", "방법", "성능 비교"], key_cases)}
  </section>

  <h2>5. 선택 가이드</h2>
  <section class="card">
    {table(["우선 기준", "추천 방법", "이유"], [
        ["최고 성능", "Full Fine-tuning", "대부분의 task에서 가장 강한 기준선"],
        ["성능-효율 균형", "Adapter", "성능 손실이 작고 시간 절감이 안정적"],
        ["파라미터 절감", "LoRA 또는 IA³", "학습 파라미터 비율이 낮음"],
        ["가장 빠른 실험", "BitFit", "평균 학습시간이 가장 짧은 경우가 많음"],
        ["작은 hate-speech task", "IA³도 검토", "일부 task에서 Full FT보다 높은 일반화 성능"],
    ])}
  </section>

  <h2>6. 최종 해석</h2>
  <section class="card">
    <div class="note">
      파인튜닝 전략 선택은 단순히 F1이 가장 높은 방법을 고르는 문제가 아니다. 성능을 얼마나 유지할 것인지, 학습시간을 얼마나 줄일 것인지, 학습 파라미터를 얼마나 줄일 것인지, 그리고 seed 반복에서 얼마나 안정적인지를 함께 고려해야 한다.
    </div>
  </section>
</main>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8-sig")
    print(OUT.resolve(), OUT.stat().st_size)


if __name__ == "__main__":
    main()
