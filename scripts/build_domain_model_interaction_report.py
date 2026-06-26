from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "study2" / "PAPER"
OUT = ROOT / "DOMAIN_MODEL_INTERACTION_REPORT.html"

MODEL_LABEL = {
    "vinai/bertweet-base": "BERTweet",
    "FacebookAI/roberta-base": "RoBERTa-base",
}
METHOD_LABEL = {
    "full_ft": "Full FT",
    "lora": "LoRA",
    "adapter": "Adapter",
    "ia3": "IA³",
    "bitfit": "BitFit",
}
METHOD_ORDER = ["full_ft", "lora", "adapter", "ia3", "bitfit"]


def esc(value) -> str:
    return html.escape(str(value))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def load_rows():
    latest = {}
    for path in RESULTS.rglob("final_metrics.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        key = (item.get("task"), item.get("model"), item.get("method"), item.get("seed"))
        if key not in latest or path.stat().st_mtime > latest[key][0]:
            latest[key] = (path.stat().st_mtime, item)
    rows = []
    for _, item in latest.values():
        rows.append(
            {
                "task": item.get("task", ""),
                "model": item.get("model", ""),
                "method": item.get("method", ""),
                "seed": item.get("seed", ""),
                "f1": float(item.get("test_macro_f1", 0) or 0),
                "accuracy": float(item.get("test_accuracy", 0) or 0),
                "seconds": float(item.get("train_seconds", 0) or 0),
                "train_rows": int(item.get("train_rows", 0) or 0),
                "test_rows": int(item.get("test_rows", 0) or 0),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"], row["method"])].append(row)
    out = {}
    for key, group in grouped.items():
        out[key] = {
            "n": len(group),
            "f1": mean([x["f1"] for x in group]),
            "sd": sd([x["f1"] for x in group]),
            "seconds": mean([x["seconds"] for x in group]),
            "train_rows": group[0]["train_rows"],
            "test_rows": group[0]["test_rows"],
        }
    return out


def table(headers, rows):
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def sign(value):
    return "+" if value >= 0 else ""


def main():
    rows = load_rows()
    summary = summarize(rows)

    tasks = sorted(set(row["task"] for row in rows))
    models = ["vinai/bertweet-base", "FacebookAI/roberta-base"]

    # Compare the two pretrained models under Full FT only.
    full_rows = []
    full_advantage = {"BERTweet": 0, "RoBERTa-base": 0, "tie": 0}
    for task in tasks:
        a = summary.get((task, "vinai/bertweet-base", "full_ft"))
        b = summary.get((task, "FacebookAI/roberta-base", "full_ft"))
        if not a or not b or a["n"] < 5 or b["n"] < 5:
            continue
        diff = a["f1"] - b["f1"]
        winner = "BERTweet" if diff > 0 else "RoBERTa-base" if diff < 0 else "동률"
        full_advantage[winner if winner != "동률" else "tie"] += 1
        cls = "pos" if diff > 0 else "neg" if diff < 0 else ""
        full_rows.append(
            [
                esc(task),
                f"{a['train_rows']:,} / {a['test_rows']:,}",
                f"{fmt(a['f1'])} ± {fmt(a['sd'])}",
                f"{fmt(b['f1'])} ± {fmt(b['sd'])}",
                f'<span class="{cls}">{sign(diff)}{fmt(diff)}</span>',
                esc(winner),
            ]
        )

    # Best method per model and task.
    interaction_rows = []
    interaction_cases = []
    for task in tasks:
        if not all(summary.get((task, model, "full_ft")) for model in models):
            continue
        task_complete_models = []
        for model in models:
            items = []
            for method in METHOD_ORDER:
                item = summary.get((task, model, method))
                if item and item["n"] >= 5:
                    items.append((method, item))
            if len(items) < 5:
                continue
            best_method, best_item = max(items, key=lambda x: x[1]["f1"])
            full_item = summary.get((task, model, "full_ft"))
            task_complete_models.append((model, best_method, best_item, full_item))
            interaction_rows.append(
                [
                    esc(task),
                    esc(MODEL_LABEL.get(model, model)),
                    esc(METHOD_LABEL.get(best_method, best_method)),
                    f"{fmt(best_item['f1'])} ± {fmt(best_item['sd'])}",
                    f"{sign(best_item['f1'] - full_item['f1'])}{fmt(best_item['f1'] - full_item['f1'])}",
                    f"{best_item['n']}/5",
                ]
            )
        if len(task_complete_models) == 2:
            # Compare the best achievable result after choosing method.
            bert = next(x for x in task_complete_models if x[0] == "vinai/bertweet-base")
            rob = next(x for x in task_complete_models if x[0] == "FacebookAI/roberta-base")
            diff = bert[2]["f1"] - rob[2]["f1"]
            winner = "BERTweet" if diff > 0 else "RoBERTa-base" if diff < 0 else "동률"
            interaction_cases.append((task, winner, diff, bert, rob))

    best_rows = []
    for task, winner, diff, bert, rob in interaction_cases:
        cls = "pos" if diff > 0 else "neg" if diff < 0 else ""
        best_rows.append(
            [
                esc(task),
                f"{METHOD_LABEL[bert[1]]} / {fmt(bert[2]['f1'])}",
                f"{METHOD_LABEL[rob[1]]} / {fmt(rob[2]['f1'])}",
                f'<span class="{cls}">{sign(diff)}{fmt(diff)}</span>',
                esc(winner),
            ]
        )

    # Cases where task is tweet-like but BERTweet does not clearly dominate.
    non_dominance = []
    for task, winner, diff, bert, rob in interaction_cases:
        if task.startswith("tweet") and winner != "BERTweet":
            non_dominance.append((task, winner, diff, bert, rob))

    # Current completion.
    completed = len(rows)

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2457d6;--green:#11845b;--red:#b42318;--amber:#9a6700}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.58}
    .wrap{max-width:1120px;margin:0 auto;padding:34px 22px 70px}.hero{background:linear-gradient(135deg,#172033,#2457d6);color:white;border-radius:24px;padding:34px;box-shadow:0 18px 45px rgba(25,40,80,.18)}
    h1{margin:0 0 10px;font-size:30px;letter-spacing:-.03em} h2{margin:34px 0 14px;font-size:22px;letter-spacing:-.02em} h3{margin:0 0 8px;font-size:18px}
    .hero p{margin:6px 0;color:#e9efff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:16px}.stat b{display:block;font-size:25px}.stat span{font-size:13px;color:#dbe5ff}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin:14px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}.note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px 16px;border-radius:12px;color:#2a3345;margin:12px 0}
    table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle} th{background:#f1f4fa;color:#465164;font-weight:700}
    .pos{color:var(--green);font-weight:800}.neg{color:var(--red);font-weight:800}.muted{color:var(--muted)}.small{font-size:13px;color:#dbe5ff}
    ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}@media(max-width:900px){.grid{grid-template-columns:1fr}.wrap{padding:20px 12px}.hero{padding:24px}table{font-size:12px}}
    """

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>도메인 특화 모델은 항상 유리한가?</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>도메인 특화 모델은 항상 유리한가?</h1>
    <p>사전학습 모델과 파인튜닝 전략의 상호작용을 Study 2 수치로 분리해 정리한 보고서입니다.</p>
    <p class="small">현재 반영된 Study 2 완료 run: {completed}/450 · 완료된 task/model/method 조합만 해석에 사용</p>
    <div class="grid">
      <div class="stat"><b>{full_advantage['BERTweet']}</b><span>Full FT 기준 BERTweet 우위 task</span></div>
      <div class="stat"><b>{full_advantage['RoBERTa-base']}</b><span>Full FT 기준 RoBERTa 우위 task</span></div>
      <div class="stat"><b>{len(interaction_cases)}</b><span>두 모델 비교 가능 task</span></div>
      <div class="stat"><b>{len(non_dominance)}</b><span>트윗 task 중 BERTweet 비우위 사례</span></div>
    </div>
  </section>

  <h2>1. 핵심 결론</h2>
  <section class="card">
    <div class="note">
      <b>결론:</b> 도메인 특화 모델인 BERTweet이 항상 유리하지는 않습니다. 
      일부 트윗 task에서는 BERTweet이 강하지만, 다른 트윗 task에서는 RoBERTa-base가 더 높거나, RoBERTa-base에 특정 파인튜닝 전략을 적용했을 때 더 좋은 결과가 나왔습니다.
    </div>
    <ul>
      <li>Full FT만 비교하면 BERTweet은 일부 트윗 데이터에서 우위지만, 모든 트윗 task에서 일관되게 우세하지 않습니다.</li>
      <li>최고 성능은 사전학습 모델 하나로 결정되지 않고, 어떤 파인튜닝 방법을 붙이는지에 따라 달라집니다.</li>
      <li>특히 tweet_hate에서는 RoBERTa-base + IA³가 BERTweet Full FT보다 높은 결과를 보이며, 모델-방법 상호작용이 강하게 나타납니다.</li>
    </ul>
  </section>

  <h2>2. Full Fine-tuning만 놓고 본 BERTweet vs RoBERTa-base</h2>
  <section class="card">
    <p class="muted">같은 Full Fine-tuning 조건에서 두 사전학습 모델의 평균 Test Macro-F1을 비교했습니다. BERTweet - RoBERTa 값이 양수면 BERTweet 우위입니다.</p>
    {table(["Task", "Train/Test 크기", "BERTweet Full FT", "RoBERTa Full FT", "차이", "우위 모델"], full_rows)}
  </section>

  <h2>3. 파인튜닝 전략까지 포함했을 때의 최고 조합</h2>
  <section class="card">
    <p class="muted">각 task/model에서 5개 방법 중 가장 높은 Macro-F1을 낸 조합입니다. 이 표는 “모델 자체”보다 “모델 × 방법” 조합이 중요하다는 점을 보여줍니다.</p>
    {table(["Task", "BERTweet 최고 조합", "RoBERTa 최고 조합", "BERTweet - RoBERTa", "최종 우위"], best_rows)}
  </section>

  <h2>4. 모델별 최고 파인튜닝 방법</h2>
  <section class="card">
    {table(["Task", "모델", "최고 방법", "Macro-F1 ± SD", "Full FT 대비", "Seed"], interaction_rows)}
  </section>

  <h2>5. 논문에 쓸 수 있는 해석 문장</h2>
  <section class="card">
    <p>
      본 연구의 Study 2 결과는 도메인 특화 사전학습 모델이 항상 최적의 선택은 아님을 보여준다. 
      BERTweet은 일부 트윗 기반 task에서 범용 RoBERTa-base보다 높은 성능을 보였지만, 모든 트윗 task에서 일관된 우위를 보이지는 않았다. 
      또한 RoBERTa-base에 IA³와 같은 특정 파인튜닝 전략을 적용했을 때 BERTweet Full Fine-tuning보다 높은 성능을 보이는 사례가 관찰되었다. 
      이는 최종 성능이 사전학습 도메인만으로 결정되지 않고, 데이터셋 특성과 파인튜닝 전략의 상호작용에 의해 결정됨을 시사한다.
    </p>
  </section>
</main>
</body>
</html>
"""

    OUT.write_text(html_doc, encoding="utf-8-sig")
    print(OUT.resolve(), OUT.stat().st_size)


if __name__ == "__main__":
    main()
