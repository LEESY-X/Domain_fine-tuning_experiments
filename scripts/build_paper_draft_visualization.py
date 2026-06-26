from __future__ import annotations

import datetime
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "PAPER_DRAFT_VISUALIZATION.html"

METHOD_ORDER = ["full_ft", "lora", "adapter", "ia3", "bitfit"]
METHOD_LABEL = {
    "full_ft": "Full FT",
    "lora": "LoRA",
    "adapter": "Adapter",
    "ia3": "IA³",
    "bitfit": "BitFit",
}
METHOD_COLOR = {
    "full_ft": "#2457d6",
    "lora": "#7c3aed",
    "adapter": "#11845b",
    "ia3": "#d97706",
    "bitfit": "#64748b",
}


def esc(value) -> str:
    return html.escape(str(value))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def pct(value):
    return f"{value * 100:.2f}%"


def minutes(seconds):
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
            latest[key] = (mtime, payload)

    rows = []
    for _, item in latest.values():
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
                "params": int(item.get("trainable_params", 0) or 0),
                "ratio": float(item.get("trainable_parameter_ratio", 0) or 0),
                "train_rows": int(item.get("train_rows", 0) or 0),
                "val_rows": int(item.get("validation_rows", 0) or 0),
                "test_rows": int(item.get("test_rows", 0) or 0),
                "epochs": int(item.get("epochs_requested", 0) or 0),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["task"], row["model"], row["method"])].append(row)
    out = []
    for (study, task, model, method), group in grouped.items():
        first = group[0]
        out.append(
            {
                "study": study,
                "task": task,
                "model": model,
                "method": method,
                "n": len(group),
                "f1": mean([x["f1"] for x in group]),
                "sd": sd([x["f1"] for x in group]),
                "accuracy": mean([x["accuracy"] for x in group]),
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
        for x in out
        if x["method"] == "full_ft"
    }
    for item in out:
        base = full.get((item["study"], item["task"], item["model"]))
        if base:
            item["delta_full"] = item["f1"] - base["f1"]
            item["time_saving"] = (
                (base["seconds"] - item["seconds"]) / base["seconds"]
                if base["seconds"]
                else 0
            )
        else:
            item["delta_full"] = 0
            item["time_saving"] = 0
    return out


def progress():
    data = {}
    for study in ["study1", "study2", "study3"]:
        data[study] = read_json(RESULTS / study / "PAPER" / "progress.json") or {}
    return data


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def bar(value, max_value=1.0, color="#2457d6", label=""):
    width = max(0, min(100, value / max_value * 100 if max_value else 0))
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>'
        f'<span>{esc(label)}</span></div>'
    )


def method_ranking(items, compact=False):
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    ranked = sorted(items, key=lambda x: (-x["f1"], order.get(x["method"], 99)))
    max_f1 = max([x["f1"] for x in ranked], default=1)
    rows = []
    for idx, item in enumerate(ranked, 1):
        method = item["method"]
        best = '<span class="rank-badge">1</span>' if idx == 1 else str(idx)
        delta = item["delta_full"]
        delta_class = "pos" if delta >= 0 else "neg"
        saving_class = "pos" if item["time_saving"] >= 0 else "neg"
        rows.append(
            [
                best,
                f'<span class="dot" style="background:{METHOD_COLOR.get(method, "#64748b")}"></span>{esc(METHOD_LABEL.get(method, method))}',
                str(item["n"]),
                bar(item["f1"], max_f1, METHOD_COLOR.get(method, "#64748b"), f"{fmt(item['f1'], 4)} ± {fmt(item['sd'], 4)}"),
                f'<span class="{delta_class}">{"+" if delta >= 0 else ""}{fmt(delta, 4)}</span>',
                f'<span class="{saving_class}">{item["time_saving"] * 100:.1f}%</span>',
                minutes(item["seconds"]),
                pct(item["ratio"]) if compact else f"{item['params']:,} / {pct(item['ratio'])}",
            ]
        )
    return table(
        ["순위", "방법", "Seed", "Macro-F1", "Full FT 대비", "시간 절감", "학습시간", "학습 파라미터"],
        rows,
    )


def generate():
    rows = load_rows()
    summary = summarize(rows)
    prog = progress()

    completed1 = int(prog.get("study1", {}).get("completed", 0) or 0)
    completed2 = int(prog.get("study2", {}).get("completed", 0) or 0)
    completed3 = int(prog.get("study3", {}).get("completed", 0) or 0)
    current = prog.get("study2", {}).get("current", {}) or {}
    total_done = completed1 + completed2 + completed3
    remaining = 550 - total_done
    remaining_study2 = 450 - completed2

    # Current news_topic-based estimate around the latest measured state.
    news_items = [
        x
        for x in summary
        if x["study"] == "study2" and x["task"] == "news_topic" and x["n"] > 0
    ]
    news_avg = mean([x["seconds"] for x in news_items]) if news_items else 490
    # If only one method/model is available, use observed recent news average.
    est_hours = remaining_study2 * news_avg / 3600 if remaining_study2 else 0
    now = datetime.datetime.now()
    finish = now + datetime.timedelta(hours=est_hours)

    completed_groups = []
    for key in sorted(set((x["study"], x["task"], x["model"]) for x in summary)):
        items = [x for x in summary if (x["study"], x["task"], x["model"]) == key]
        if len(items) >= 5 and all(x["n"] >= 5 for x in items):
            completed_groups.append((key, items))

    best_counts = defaultdict(int)
    for _, items in completed_groups:
        best = max(items, key=lambda x: x["f1"])
        best_counts[METHOD_LABEL.get(best["method"], best["method"])] += 1

    # Study 1/3 result cards.
    fixed_result_cards = []
    for study in ["study1", "study3"]:
        for key, items in completed_groups:
            if key[0] != study:
                continue
            first = items[0]
            fixed_result_cards.append(
                f"""
                <section class="card result-card">
                  <h3>{esc(key[0].upper())} · {esc(key[1])}</h3>
                  <p class="meta">{esc(key[2])} · train {first['train_rows']:,} / val {first['val_rows']:,} / test {first['test_rows']:,} · Epoch {first['epochs']}</p>
                  {method_ranking(items, compact=True)}
                </section>
                """
            )

    # Study 2 currently completed task/model groups.
    study2_cards = []
    for key, items in completed_groups:
        if key[0] != "study2":
            continue
        first = items[0]
        study2_cards.append(
            f"""
            <section class="card compact">
              <h3>{esc(key[1])} · {esc(key[2])}</h3>
              <p class="meta">train {first['train_rows']:,} / val {first['val_rows']:,} / test {first['test_rows']:,} · Epoch {first['epochs']}</p>
              {method_ranking(items, compact=True)}
            </section>
            """
        )

    # Domain model interaction summary for finished comparable tasks.
    interaction_rows = []
    for task in sorted(set(x["task"] for x in summary if x["study"] == "study2")):
        bert = [x for x in summary if x["study"] == "study2" and x["task"] == task and x["model"] == "vinai/bertweet-base" and x["n"] >= 5]
        rob = [x for x in summary if x["study"] == "study2" and x["task"] == task and x["model"] == "FacebookAI/roberta-base" and x["n"] >= 5]
        if len(bert) < 5 or len(rob) < 5:
            continue
        bert_best = max(bert, key=lambda x: x["f1"])
        rob_best = max(rob, key=lambda x: x["f1"])
        diff = bert_best["f1"] - rob_best["f1"]
        winner = "BERTweet" if diff > 0 else "RoBERTa-base" if diff < 0 else "동률"
        cls = "pos" if diff > 0 else "neg" if diff < 0 else ""
        interaction_rows.append(
            [
                esc(task),
                f"{METHOD_LABEL[bert_best['method']]} / {fmt(bert_best['f1'], 4)}",
                f"{METHOD_LABEL[rob_best['method']]} / {fmt(rob_best['f1'], 4)}",
                f'<span class="{cls}">{"+" if diff >= 0 else ""}{fmt(diff, 4)}</span>',
                esc(winner),
            ]
        )

    best_count_text = ", ".join(f"{k} {v}회" for k, v in sorted(best_counts.items(), key=lambda x: -x[1]))

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2457d6;--green:#11845b;--red:#b42318;--amber:#d97706;--purple:#7c3aed}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.58}
    .wrap{max-width:1240px;margin:0 auto;padding:34px 22px 80px}.hero{background:linear-gradient(135deg,#111827,#2457d6);color:white;border-radius:26px;padding:36px;box-shadow:0 20px 50px rgba(25,40,80,.20)}
    h1{margin:0 0 10px;font-size:34px;letter-spacing:-.04em} h2{margin:34px 0 14px;font-size:23px;letter-spacing:-.03em} h3{margin:0 0 8px;font-size:18px;letter-spacing:-.02em}.hero p{margin:7px 0;color:#e9efff}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:16px}.stat b{display:block;font-size:27px}.stat span{font-size:13px;color:#dbe5ff}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin:14px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}.compact{padding:18px}.result-card{overflow:auto}.meta{color:var(--muted);font-size:13px;margin:4px 0 14px}
    .pill{display:inline-block;background:#eef4ff;color:#2457d6;border:1px solid #d7e4ff;border-radius:999px;padding:5px 10px;font-size:12px;font-weight:800;margin:4px 6px 4px 0}.note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px 16px;border-radius:12px;color:#2a3345;margin:12px 0}
    .flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.step{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px;text-align:center}.step b{display:block;color:#2457d6;margin-bottom:4px}
    table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle} th{background:#f1f4fa;color:#465164;font-weight:700}
    .bar{position:relative;height:24px;background:#eef2f7;border-radius:999px;overflow:hidden;min-width:155px}.bar-fill{height:100%;border-radius:999px}.bar span{position:absolute;inset:0;display:flex;align-items:center;padding-left:9px;font-size:12px;font-weight:800;color:#111827}.rank-badge{display:inline-block;background:#eaf1ff;color:#2457d6;font-weight:900;padding:2px 8px;border-radius:999px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.pos{color:var(--green);font-weight:800}.neg{color:var(--red);font-weight:800}.small{font-size:13px;color:#dbe5ff} ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}
    .paperbox{border:1px solid #c7d2fe;background:#f8fbff;border-radius:18px;padding:18px}.section-title{font-weight:900;color:#111827;margin-bottom:6px}
    @media(max-width:980px){.grid,.grid2,.grid3,.flow{grid-template-columns:1fr}.wrap{padding:20px 12px}.hero{padding:24px}table{font-size:12px}.bar{min-width:120px}}
    """

    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>논문 초안 시각화: Fine-tuning Strategy Study</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>논문 초안 시각화</h1>
    <p>Full Fine-tuning과 PEFT 방법을 <b>성능·효율·안정성</b>이라는 3가지 지표로 비교하는 연구 초안입니다.</p>
    <p class="small">생성 시각: {now.strftime('%Y-%m-%d %H:%M:%S')} KST · 현재 완료 기준 자동 집계</p>
    <div class="grid">
      <div class="stat"><b>{total_done}/550</b><span>전체 완료 run</span></div>
      <div class="stat"><b>{completed2}/450</b><span>Study 2 완료 run</span></div>
      <div class="stat"><b>{remaining}</b><span>전체 남은 run</span></div>
      <div class="stat"><b>{est_hours:.1f}h</b><span>잔여 추정 · 종료 {finish.strftime('%H:%M')} 전후</span></div>
    </div>
  </section>

  <h2>1. 논문 제목과 핵심 주장</h2>
  <section class="card paperbox">
    <div class="section-title">추천 제목</div>
    <h3>최고의 파인튜닝은 없다: 도메인을 가로지르는 성능·안정성·효율성의 트레이드오프</h3>
    <p class="meta">There Is No Best Fine-Tuning Strategy: Performance, Stability, and Efficiency Trade-offs Across Domains</p>
    <div class="note">
      <b>핵심 주장:</b> 최적의 파인튜닝 방법은 고정되어 있지 않다. 본 연구는 각 방법을 단일 F1 점수가 아니라 <b>성능(Performance), 효율(Efficiency), 안정성(Stability)</b>의 3가지 지표로 평가한다. Full FT는 강한 성능 기준선이지만, Adapter·LoRA·IA³·BitFit은 데이터셋의 도메인, 크기, 분포, 사전학습 모델 적합성에 따라 서로 다른 장단점을 보인다.
    </div>
  </section>

  <h2>2. 평가 프레임워크: 3가지 핵심 지표</h2>
  <section class="grid3">
    <div class="card">
      <h3>성능 Performance</h3>
      <p>모델이 실제 분류 과제를 얼마나 잘 해결하는지를 평가합니다.</p>
      <span class="pill">Test Macro-F1</span>
      <span class="pill">Accuracy</span>
      <div class="note">주요 판단 기준은 class imbalance에 비교적 강한 <b>Test Macro-F1</b>입니다. Accuracy는 보조 지표로 사용합니다.</div>
    </div>
    <div class="card">
      <h3>효율 Efficiency</h3>
      <p>같은 성능을 얻기 위해 필요한 학습 비용을 평가합니다.</p>
      <span class="pill">Trainable Parameters</span>
      <span class="pill">Parameter Ratio</span>
      <span class="pill">Training Time</span>
      <div class="note">PEFT가 파라미터 수는 줄이지만 실제 wall-clock time까지 항상 줄이는지는 별도로 검증합니다.</div>
    </div>
    <div class="card">
      <h3>안정성 Stability</h3>
      <p>한 번의 우연한 결과가 아니라 반복 실행에서도 일관적인지를 평가합니다.</p>
      <span class="pill">5 Seeds</span>
      <span class="pill">Standard Deviation</span>
      <div class="note">각 조합을 5개 seed로 반복하고, 평균 성능과 표준편차를 함께 보고 방법의 신뢰성을 판단합니다.</div>
    </div>
  </section>

  <h2>3. 연구 질문</h2>
  <section class="grid2">
    <div class="card"><h3>RQ1. 성능 관점에서 Full FT가 항상 최고인가?</h3><p>동일 조건에서 Full FT와 PEFT의 Test Macro-F1 및 Accuracy를 비교한다.</p></div>
    <div class="card"><h3>RQ2. 효율 관점에서 PEFT는 항상 유리한가?</h3><p>LoRA/Adapter/IA³/BitFit의 trainable parameter ratio와 실제 wall-clock training time을 함께 본다.</p></div>
    <div class="card"><h3>RQ3. 안정성 관점에서 어떤 방법이 일관적인가?</h3><p>5개 seed 반복의 평균과 표준편차를 통해 특정 방법의 결과가 우연인지 확인한다.</p></div>
    <div class="card"><h3>RQ4. 도메인 특화 모델은 항상 유리한가?</h3><p>BERTweet과 RoBERTa-base를 비교하여 사전학습 도메인과 파인튜닝 전략의 상호작용을 분석한다.</p></div>
  </section>

  <h2>4. 실험 설계</h2>
  <section class="card">
    <div class="flow">
      <div class="step"><b>3 Studies</b><span>트윗 혐오표현 · 영어 다중 도메인 · 한국어 데이터</span></div>
      <div class="step"><b>5 Methods</b><span>Full FT, LoRA, Adapter, IA³, BitFit</span></div>
      <div class="step"><b>5 Seeds</b><span>42, 52, 62, 72, 82</span></div>
      <div class="step"><b>550 Runs</b><span>동일 평가 체계 반복</span></div>
      <div class="step"><b>3 Metrics</b><span>성능 · 효율 · 안정성</span></div>
    </div>
  </section>
  <section class="card">
    {table(["Study", "모델", "데이터셋", "Epoch", "Run 수", "상태"], [
        ["Study 1", "vinai/bertweet-base", "measuring_hate_speech", "3", "25", f"완료 {completed1}/25"],
        ["Study 2", "vinai/bertweet-base, FacebookAI/roberta-base", "9개 영어 task", "2", "450", f"진행 중 {completed2}/450"],
        ["Study 3", "klue/roberta-base", "YNAT, NSMC, K-MHaS", "5", "75", f"완료 {completed3}/75"],
    ])}
  </section>

  <h2>5. 현재 결과 요약</h2>
  <section class="grid3">
    <div class="card"><h3>완료 그룹 1위 분포</h3><p>{esc(best_count_text)}</p><p class="meta">5개 방법과 5개 seed가 모두 완료된 task/model 조합 기준</p></div>
    <div class="card"><h3>Study 3 핵심</h3><p>YNAT에서는 Adapter가 Full FT를 넘었고, NSMC/K-MHaS에서는 Full FT가 가장 강했다.</p></div>
    <div class="card"><h3>Study 2 핵심</h3><p>영어 다중 도메인에서는 Full FT 우위가 강하지만, tweet_hate에서 IA³가 예외적으로 강하게 나타났다.</p></div>
  </section>

  <h2>6. 3가지 지표 기준 현재 해석</h2>
  <section class="grid3">
    <div class="card">
      <h3>성능 기준</h3>
      <p>현재까지 Full FT는 다수의 task에서 가장 높은 Macro-F1을 보이며 강한 기준선으로 작동합니다. 다만 YNAT의 Adapter, tweet_hate의 IA³처럼 PEFT가 Full FT를 넘는 예외도 확인됩니다.</p>
    </div>
    <div class="card">
      <h3>효율 기준</h3>
      <p>Adapter, LoRA, IA³, BitFit은 Full FT보다 훨씬 적은 학습 파라미터를 사용합니다. 그러나 LoRA처럼 파라미터 효율은 높아도 실제 학습시간이 항상 짧지는 않은 경우가 있어, 효율은 파라미터와 시간으로 분리해 해석해야 합니다.</p>
    </div>
    <div class="card">
      <h3>안정성 기준</h3>
      <p>5개 seed 반복을 통해 평균뿐 아니라 표준편차를 확인합니다. 표준편차가 낮은 방법은 재현성이 좋지만, 낮은 분산이 반드시 높은 성능을 의미하지는 않으므로 성능과 함께 판단합니다.</p>
    </div>
  </section>

  <h2>7. Study 1/3 확정 결과</h2>
  {''.join(fixed_result_cards)}

  <h2>8. Study 2 중간 결과</h2>
  <section class="card">
    <p>Study 2는 아직 진행 중입니다. 현재는 마지막 <b>news_topic</b> 구간을 돌고 있으며, 전체 결론은 450/450 완료 후 확정하는 것이 안전합니다.</p>
    <p class="meta">현재 실행 중: {esc(current.get('task_key', ''))} / {esc(current.get('model_name', ''))} / {esc(current.get('method', ''))} / seed {esc(current.get('seed', ''))}</p>
  </section>
  {''.join(study2_cards)}

  <h2>9. 도메인 특화 모델과 전략 상호작용</h2>
  <section class="card">
    <p>아래 표는 각 task에서 BERTweet과 RoBERTa-base가 “각자 가장 좋은 파인튜닝 방법”을 선택했을 때의 비교입니다.</p>
    {table(["Task", "BERTweet 최고 조합", "RoBERTa 최고 조합", "BERTweet - RoBERTa", "우위"], interaction_rows)}
    <div class="note">
      <b>해석:</b> 도메인 특화 모델이 항상 유리하지는 않다. 트윗 데이터에서도 RoBERTa-base + 특정 PEFT 조합이 더 나은 경우가 있으며, 이는 사전학습 도메인보다 모델 × 파인튜닝 전략 × 데이터셋 특성의 상호작용이 중요함을 보여준다.
    </div>
  </section>

  <h2>10. 논문 초안 구조</h2>
  <section class="grid2">
    <div class="card"><h3>Introduction</h3><ul><li>Full FT는 강하지만 비용이 큼</li><li>PEFT는 효율적이라고 알려져 있으나 실제 선택 기준은 불명확</li><li>본 연구는 성능·효율·안정성의 3가지 지표로 fine-tuning strategy를 평가함</li></ul></div>
    <div class="card"><h3>Related Work</h3><ul><li>Adapter, LoRA, BitFit, IA³</li><li>BERTweet과 RoBERTa</li><li>PEFT benchmark와 효율성 평가 연구</li></ul></div>
    <div class="card"><h3>Method</h3><ul><li>3 studies, 5 methods, 5 seeds</li><li>동일 batch/length/precision 조건</li><li>성능: Macro-F1/Accuracy</li><li>효율: training time/trainable params</li><li>안정성: seed 평균/표준편차</li></ul></div>
    <div class="card"><h3>Discussion</h3><ul><li>최고 방법은 데이터셋별로 달라짐</li><li>파라미터 효율 ≠ 시간 효율</li><li>안정적인 방법이 항상 최고 성능은 아님</li><li>도메인 특화 모델 우위는 fine-tuning strategy에 의해 뒤집힐 수 있음</li></ul></div>
  </section>

  <h2>11. 결론 초안</h2>
  <section class="card paperbox">
    <p>
      본 연구는 동일한 모델과 데이터셋 조건에서도 파인튜닝 전략에 따라 <b>성능, 효율, 안정성</b>이 크게 달라짐을 보였다.
      Full Fine-tuning은 전반적으로 강한 기준선으로 작동했지만, Adapter와 LoRA는 훨씬 적은 학습 파라미터로 경쟁력 있는 성능을 유지했으며, IA³는 일부 hate-speech task에서 Full Fine-tuning보다 더 나은 일반화 성능을 보였다.
      특히 도메인 특화 모델이 항상 최적의 선택은 아니며, parameter-efficient 방법이 반드시 time-efficient하지도 않다는 점을 확인했다.
      따라서 실제 응용 환경에서는 단일 F1 점수만을 기준으로 파인튜닝 방법을 선택하기보다, <b>성능·효율·안정성</b>이라는 3가지 판단 기준을 함께 고려해야 한다.
    </p>
  </section>
</main>
</body>
</html>
"""

    OUT.write_text(doc, encoding="utf-8-sig")
    print(OUT.resolve(), OUT.stat().st_size)


if __name__ == "__main__":
    generate()
