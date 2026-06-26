from __future__ import annotations

import datetime as dt
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "FINAL_PAPER_ANALYSIS_REPORT.html"

METHOD_ORDER = ["full_ft", "lora", "adapter", "ia3", "bitfit"]
METHOD_LABEL = {
    "full_ft": "Full Fine-tuning",
    "lora": "LoRA",
    "adapter": "Adapter",
    "ia3": "IA³",
    "bitfit": "BitFit",
}
METHOD_COLOR = {
    "full_ft": "#2563eb",
    "lora": "#7c3aed",
    "adapter": "#059669",
    "ia3": "#d97706",
    "bitfit": "#64748b",
}

TASK_DESCRIPTION = {
    "measuring_hate_speech": "영어 소셜 미디어 댓글 기반 혐오표현/유해성 분류 데이터셋이다. Study 1에서는 BERTweet 기반 단일 도메인 심층 비교에 사용했다.",
    "tweet_sentiment": "TweetEval sentiment task로, 짧은 트윗 문장의 감성 극성을 분류한다. 트윗 특화 모델과 범용 모델의 차이를 보기 좋은 대표 트윗 task다.",
    "finance_sentiment": "FinancialPhraseBank 기반 금융 문장 감성 분류 데이터셋이다. 트윗이 아닌 전문 도메인 문장에 대한 전이 성능을 확인한다.",
    "movie_reviews": "IMDB 영화 리뷰 감성 분류 데이터셋이다. 상대적으로 긴 영어 리뷰 문장을 포함해 트윗과 다른 텍스트 길이/문체 조건을 제공한다.",
    "product_reviews": "Amazon product review 기반 다중 클래스 상품 리뷰 분류 데이터셋이다. Study 2에서 가장 큰 영어 데이터셋 중 하나로 시간 효율 차이가 잘 드러난다.",
    "tweet_emotion": "TweetEval emotion task로, 트윗에 드러난 감정 범주를 분류한다. 단순 감성보다 클래스 구분이 더 복잡하다.",
    "tweet_hate": "TweetEval hate task로, 트윗 기반 혐오표현 여부를 분류한다. 작은 데이터에서 Full FT와 PEFT의 일반화 차이가 크게 관찰된 핵심 예외 사례다.",
    "tweet_offensive": "TweetEval offensive task로, 공격적 표현 여부를 분류한다. 트윗 특화 모델과 전략별 성능 손실을 비교한다.",
    "tweet_irony": "TweetEval irony task로, 트윗의 아이러니 여부를 분류한다. 작은 데이터셋에서 안정성 지표를 보기 좋다.",
    "news_topic": "AG News 기반 영어 뉴스 주제 분류 데이터셋이다. Study 2의 마지막 대규모 일반 도메인 task로, 트윗 도메인 밖에서 BERTweet과 RoBERTa를 비교한다.",
    "news_ynat": "KLUE YNAT 한국어 뉴스 제목 주제 분류 데이터셋이다. Study 3에서 Adapter가 Full FT보다 높은 성능을 보인 핵심 사례다.",
    "movie_nsmc": "NSMC 한국어 영화 리뷰 감성 분류 데이터셋이다. 큰 규모의 한국어 감성 task에서 Full FT의 강점을 확인한다.",
    "comment_kmhas_binary": "K-MHaS 기반 한국어 혐오/비혐오 이진 분류 데이터셋이다. 한국어 혐오표현 영역에서 전략별 성능 손실과 안정성을 비교한다.",
}

METHOD_DESCRIPTION = {
    "full_ft": "모든 모델 파라미터를 업데이트하는 표준 방식이다. 가장 강한 성능 기준선이지만 학습 파라미터와 저장 비용이 가장 크다.",
    "lora": "기존 weight를 고정하고 attention projection 등에 low-rank matrix를 추가로 학습하는 방식이다. 파라미터 효율은 높지만 실제 학습시간이 항상 짧지는 않았다.",
    "adapter": "Transformer 내부에 작은 bottleneck module을 삽입해 해당 모듈 중심으로 학습하는 방식이다. 본 실험에서는 성능과 시간 절감의 균형이 가장 좋은 후보 중 하나였다.",
    "ia3": "activation/channel scaling 계수를 학습하는 매우 가벼운 PEFT 방식이다. 평균 성능은 낮은 편이지만 일부 hate-speech task에서 Full FT를 넘는 예외적 결과를 보였다.",
    "bitfit": "bias parameter만 업데이트하는 초경량 방식이다. 가장 단순하고 빠른 축에 속하지만, 복잡한 task에서는 성능 손실이 커질 수 있다.",
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def esc(x) -> str:
    return html.escape(str(x))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def fmt(x, n=4):
    return f"{x:.{n}f}"


def pct(x):
    return f"{x * 100:.2f}%"


def minutes(seconds):
    return f"{seconds / 60:.1f}분"


def signed(x, n=4):
    return f"{'+' if x >= 0 else ''}{x:.{n}f}"


def load_rows():
    latest = {}
    for path in RESULTS.rglob("final_metrics.json"):
        item = read_json(path)
        if not item:
            continue
        key = (
            item.get("study"),
            item.get("task"),
            item.get("model"),
            item.get("method"),
            item.get("seed"),
        )
        if key not in latest or path.stat().st_mtime > latest[key][0]:
            latest[key] = (path.stat().st_mtime, item)

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
                "completed_at": item.get("completed_at", ""),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["task"], row["model"], row["method"])].append(row)

    summary = []
    for (study, task, model, method), group in grouped.items():
        first = group[0]
        summary.append(
            {
                "study": study,
                "task": task,
                "model": model,
                "method": method,
                "n": len(group),
                "f1": mean([x["f1"] for x in group]),
                "f1_sd": stdev([x["f1"] for x in group]),
                "accuracy": mean([x["accuracy"] for x in group]),
                "precision": mean([x["precision"] for x in group]),
                "recall": mean([x["recall"] for x in group]),
                "seconds": mean([x["seconds"] for x in group]),
                "seconds_sd": stdev([x["seconds"] for x in group]),
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
        for x in summary
        if x["method"] == "full_ft"
    }
    for item in summary:
        base = full.get((item["study"], item["task"], item["model"]))
        if base:
            item["delta_full"] = item["f1"] - base["f1"]
            item["time_saving"] = (
                (base["seconds"] - item["seconds"]) / base["seconds"]
                if base["seconds"]
                else 0.0
            )
        else:
            item["delta_full"] = 0.0
            item["time_saving"] = 0.0
    return summary


def progress():
    return {
        s: read_json(RESULTS / s / "PAPER" / "progress.json") or {}
        for s in ["study1", "study2", "study3"]
    }


def table(headers, rows, cls=""):
    klass = f' class="{cls}"' if cls else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table{klass}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def bar(value, max_value, color, label):
    width = 0 if max_value == 0 else max(0, min(100, value / max_value * 100))
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>'
        f"<span>{esc(label)}</span></div>"
    )


def completed_groups(summary):
    groups = []
    for key in sorted(set((x["study"], x["task"], x["model"]) for x in summary)):
        items = [x for x in summary if (x["study"], x["task"], x["model"]) == key]
        if len(items) == 5 and all(x["n"] == 5 for x in items):
            groups.append((key, items))
    return groups


def ranking_table(items):
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    ranked = sorted(items, key=lambda x: (-x["f1"], order.get(x["method"], 99)))
    max_f1 = max(x["f1"] for x in ranked)
    rows = []
    for rank, item in enumerate(ranked, 1):
        method = item["method"]
        color = METHOD_COLOR[method]
        delta_cls = "pos" if item["delta_full"] >= 0 else "neg"
        saving_cls = "pos" if item["time_saving"] >= 0 else "neg"
        rank_cell = '<span class="rank">1</span>' if rank == 1 else str(rank)
        rows.append(
            [
                rank_cell,
                f'<span class="dot" style="background:{color}"></span>{METHOD_LABEL[method]}',
                str(item["n"]),
                bar(item["f1"], max_f1, color, f"{fmt(item['f1'])} ± {fmt(item['f1_sd'])}"),
                f'<span class="{delta_cls}">{signed(item["delta_full"])}</span>',
                f'<span class="{saving_cls}">{item["time_saving"] * 100:.1f}%</span>',
                minutes(item["seconds"]),
                f"{item['params']:,}",
                pct(item["ratio"]),
            ]
        )
    return table(
        [
            "순위",
            "방법",
            "Seed",
            "성능: Macro-F1 ± SD",
            "Full FT 대비 F1",
            "효율: 시간 절감",
            "평균 학습시간",
            "학습 파라미터",
            "파라미터 비율",
        ],
        rows,
    )


def count_text(counter):
    return ", ".join(f"{k} {v}회" for k, v in sorted(counter.items(), key=lambda x: -x[1]))


def make_report():
    rows = load_rows()
    summary = summarize(rows)
    prog = progress()
    groups = completed_groups(summary)
    now = dt.datetime.now()
    config = read_json(ROOT / "config" / "experiment_config.json") or {}
    env = read_json(RESULTS / "environment.json") or {}

    completed1 = int(prog["study1"].get("completed", 0) or 0)
    completed2 = int(prog["study2"].get("completed", 0) or 0)
    completed3 = int(prog["study3"].get("completed", 0) or 0)
    total_done = completed1 + completed2 + completed3
    failed = sum(int(prog[s].get("failed", 0) or 0) for s in ["study1", "study2", "study3"])

    total_train_hours = sum(r["seconds"] for r in rows) / 3600
    epoch_examples = sum(r["train_rows"] * r["epochs"] for r in rows)
    raw_train_rows = sum(r["train_rows"] for r in rows)
    test_rows = sum(r["test_rows"] for r in rows)

    performance_wins = defaultdict(int)
    time_efficiency_wins = defaultdict(int)
    parameter_efficiency_wins = defaultdict(int)
    stability_wins = defaultdict(int)
    indicator_rows = []
    for _, items in groups:
        best_perf = max(items, key=lambda x: x["f1"])
        best_time = min(items, key=lambda x: x["seconds"])
        best_param = min(items, key=lambda x: x["ratio"])
        best_stable = min(items, key=lambda x: x["f1_sd"])
        performance_wins[METHOD_LABEL[best_perf["method"]]] += 1
        time_efficiency_wins[METHOD_LABEL[best_time["method"]]] += 1
        parameter_efficiency_wins[METHOD_LABEL[best_param["method"]]] += 1
        stability_wins[METHOD_LABEL[best_stable["method"]]] += 1
        indicator_rows.append(
            [
                items[0]["study"],
                items[0]["task"],
                items[0]["model"],
                f"{METHOD_LABEL[best_perf['method']]} ({fmt(best_perf['f1'])})",
                f"{METHOD_LABEL[best_time['method']]} ({minutes(best_time['seconds'])})",
                f"{METHOD_LABEL[best_param['method']]} ({pct(best_param['ratio'])})",
                f"{METHOD_LABEL[best_stable['method']]} (SD {fmt(best_stable['f1_sd'])})",
            ]
        )

    method_agg_rows = []
    for method in METHOD_ORDER:
        items = [x for x in summary if x["method"] == method and x["n"] == 5]
        ranks = []
        for _, group_items in groups:
            ranked = sorted(group_items, key=lambda x: -x["f1"])
            for idx, item in enumerate(ranked, 1):
                if item["method"] == method:
                    ranks.append(idx)
                    break
        method_agg_rows.append(
            {
                "method": method,
                "groups": len(items),
                "mean_f1": mean([x["f1"] for x in items]),
                "mean_delta": mean([x["delta_full"] for x in items]),
                "mean_time_saving": mean([x["time_saving"] for x in items]),
                "mean_ratio": mean([x["ratio"] for x in items]),
                "mean_sd": mean([x["f1_sd"] for x in items]),
                "mean_rank": mean(ranks),
            }
        )

    def method_label_cell(method: str) -> str:
        return f'<span class="dot" style="background:{METHOD_COLOR[method]}"></span>{METHOD_LABEL[method]}'

    indicator_winner_table = table(
        [
            "Study",
            "Task",
            "Model",
            "성능 1위",
            "시간 효율 1위",
            "파라미터 효율 1위",
            "안정성 1위",
        ],
        indicator_rows,
    )

    method_agg_table = table(
        [
            "방법",
            "완료 그룹",
            "평균 Macro-F1",
            "평균 순위",
            "Full FT 대비 평균 F1",
                "평균 시간 절감",
                "평균 파라미터 비율",
                "평균 표준편차",
        ],
        [
            [
                method_label_cell(r["method"]),
                str(r["groups"]),
                fmt(r["mean_f1"]),
                fmt(r["mean_rank"], 2),
                f'<span class="{"pos" if r["mean_delta"] >= 0 else "neg"}">{signed(r["mean_delta"])}</span>',
                f'<span class="{"pos" if r["mean_time_saving"] >= 0 else "neg"}">{r["mean_time_saving"] * 100:.1f}%</span>',
                pct(r["mean_ratio"]),
                fmt(r["mean_sd"]),
            ]
            for r in method_agg_rows
        ],
    )

    method_intro_table = table(
        ["방법", "핵심 아이디어", "해석"],
        [
            [
                f'<span class="dot" style="background:{METHOD_COLOR[m]}"></span>{METHOD_LABEL[m]}',
                METHOD_DESCRIPTION[m],
                {
                    "full_ft": "최고 성능 기준선",
                    "lora": "파라미터 효율 검증 대상",
                    "adapter": "성능-효율 균형 후보",
                    "ia3": "초경량 PEFT와 예외적 일반화 사례",
                    "bitfit": "가장 단순한 초경량 기준선",
                }[m],
            ]
            for m in METHOD_ORDER
        ],
    )

    study_overview = table(
        ["Study", "모델", "데이터셋", "Epoch", "Run", "상태"],
        [
            ["Study 1", "vinai/bertweet-base", "measuring_hate_speech", "3", "25", f"완료 {completed1}/25"],
            [
                "Study 2",
                "vinai/bertweet-base, FacebookAI/roberta-base",
                "영어 9개 task",
                "2",
                "450",
                f"완료 {completed2}/450",
            ],
            ["Study 3", "klue/roberta-base", "YNAT, NSMC, K-MHaS", "5", "75", f"완료 {completed3}/75"],
        ],
    )

    dataset_rows = []
    seen = set()
    for item in sorted(summary, key=lambda x: (x["study"], x["task"], x["model"])):
        key = (item["study"], item["task"], item["model"])
        if key in seen:
            continue
        seen.add(key)
        dataset_rows.append(
            [
                item["study"],
                item["task"],
                item["model"],
                f"{item['train_rows']:,}",
                f"{item['val_rows']:,}",
                f"{item['test_rows']:,}",
                str(item["epochs"]),
                TASK_DESCRIPTION.get(item["task"], ""),
            ]
        )
    dataset_table = table(
        ["Study", "Task", "Model", "Train", "Validation", "Test", "Epoch", "데이터셋 설명"],
        dataset_rows,
    )

    # Study-level aggregated view.
    study_rows = []
    for study in ["study1", "study2", "study3"]:
        s_items = [x for x in summary if x["study"] == study and x["n"] == 5]
        for method in METHOD_ORDER:
            m_items = [x for x in s_items if x["method"] == method]
            if not m_items:
                continue
            study_rows.append(
                [
                    study,
                    METHOD_LABEL[method],
                    str(len(m_items)),
                    fmt(mean([x["f1"] for x in m_items])),
                    fmt(mean([x["f1_sd"] for x in m_items])),
                    f"{mean([x['time_saving'] for x in m_items]) * 100:.1f}%",
                    pct(mean([x["ratio"] for x in m_items])),
                    minutes(mean([x["seconds"] for x in m_items])),
                ]
            )
    study_agg_table = table(
        ["Study", "방법", "task/model 그룹", "평균 F1", "평균 F1 SD", "평균 시간 절감", "평균 파라미터 비율", "평균 학습시간"],
        study_rows,
    )

    # Notable cases.
    notable = []
    for key, items in groups:
        full = next(x for x in items if x["method"] == "full_ft")
        best = max(items, key=lambda x: x["f1"])
        best_peft = max([x for x in items if x["method"] != "full_ft"], key=lambda x: x["f1"])
        if best["method"] != "full_ft":
            notable.append(
                [
                    key[0],
                    key[1],
                    key[2],
                    f"{METHOD_LABEL[best['method']]}",
                    f"{fmt(best['f1'])} vs Full FT {fmt(full['f1'])}",
                    f"{signed(best['f1'] - full['f1'])}",
                    "PEFT가 Full FT를 초과한 핵심 사례",
                ]
            )
        elif best_peft["f1"] >= full["f1"] - 0.003:
            notable.append(
                [
                    key[0],
                    key[1],
                    key[2],
                    f"{METHOD_LABEL[best_peft['method']]}",
                    f"{fmt(best_peft['f1'])} vs Full FT {fmt(full['f1'])}",
                    f"{signed(best_peft['f1'] - full['f1'])}",
                    "PEFT가 Full FT에 거의 근접한 사례",
                ]
            )
    notable_table = table(
        ["Study", "Task", "Model", "주목 방법", "성능 비교", "ΔF1", "해석"],
        notable,
    )

    interaction_rows = []
    for task in sorted(set(x["task"] for x in summary if x["study"] == "study2")):
        bert = [
            x
            for x in summary
            if x["study"] == "study2"
            and x["task"] == task
            and x["model"] == "vinai/bertweet-base"
            and x["n"] == 5
        ]
        rob = [
            x
            for x in summary
            if x["study"] == "study2"
            and x["task"] == task
            and x["model"] == "FacebookAI/roberta-base"
            and x["n"] == 5
        ]
        if len(bert) != 5 or len(rob) != 5:
            continue
        bert_best = max(bert, key=lambda x: x["f1"])
        rob_best = max(rob, key=lambda x: x["f1"])
        diff = bert_best["f1"] - rob_best["f1"]
        winner = "BERTweet" if diff > 0 else "RoBERTa-base" if diff < 0 else "동률"
        cls = "pos" if diff > 0 else "neg" if diff < 0 else ""
        interaction_rows.append(
            [
                task,
                f"{METHOD_LABEL[bert_best['method']]} / {fmt(bert_best['f1'])}",
                f"{METHOD_LABEL[rob_best['method']]} / {fmt(rob_best['f1'])}",
                f'<span class="{cls}">{signed(diff)}</span>',
                winner,
            ]
        )
    interaction_table = table(
        ["Task", "BERTweet 최고 조합", "RoBERTa-base 최고 조합", "BERTweet - RoBERTa", "우위"],
        interaction_rows,
    )

    cards = []
    for key, items in groups:
        first = items[0]
        cards.append(
            f"""
            <section class="card result-card">
              <h3>{esc(key[0].upper())} · {esc(key[1])}</h3>
              <p class="meta">{esc(key[2])} · train {first['train_rows']:,} / validation {first['val_rows']:,} / test {first['test_rows']:,} · Epoch {first['epochs']}</p>
              {ranking_table(items)}
            </section>
            """
        )

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2563eb;--green:#059669;--red:#b42318;--amber:#d97706}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.58}
    .wrap{max-width:1280px;margin:0 auto;padding:34px 22px 80px}.hero{background:linear-gradient(135deg,#111827,#2563eb);color:white;border-radius:28px;padding:38px;box-shadow:0 20px 52px rgba(25,40,80,.22)}
    h1{margin:0 0 10px;font-size:35px;letter-spacing:-.04em} h2{margin:36px 0 14px;font-size:24px;letter-spacing:-.03em} h3{margin:0 0 8px;font-size:18px;letter-spacing:-.02em}.hero p{margin:7px 0;color:#e9efff}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
    .stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:16px}.stat b{display:block;font-size:27px}.stat span{font-size:13px;color:#dbe5ff}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin:14px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}.result-card{overflow:auto}.meta{color:var(--muted);font-size:13px;margin:4px 0 14px}
    .note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px 16px;border-radius:12px;color:#2a3345;margin:12px 0}.warn{border-left-color:var(--amber)}.paperbox{border:1px solid #c7d2fe;background:#f8fbff}
    .pill{display:inline-block;background:#eef4ff;color:#2457d6;border:1px solid #d7e4ff;border-radius:999px;padding:5px 10px;font-size:12px;font-weight:800;margin:4px 6px 4px 0}.step{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px;text-align:center}.step b{display:block;color:#2563eb;margin-bottom:4px}
    table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle} th{background:#f1f4fa;color:#465164;font-weight:700} tr:hover td{background:#fafcff}
    .bar{position:relative;height:24px;background:#eef2f7;border-radius:999px;overflow:hidden;min-width:160px}.bar-fill{height:100%;border-radius:999px}.bar span{position:absolute;inset:0;display:flex;align-items:center;padding-left:9px;font-size:12px;font-weight:800;color:#111827}
    .rank{display:inline-block;background:#eaf1ff;color:#2563eb;font-weight:900;padding:2px 8px;border-radius:999px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.pos{color:var(--green);font-weight:800}.neg{color:var(--red);font-weight:800}.small{font-size:13px;color:#dbe5ff} ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}
    @media(max-width:980px){.grid,.grid2,.grid3,.flow{grid-template-columns:1fr}.wrap{padding:20px 12px}.hero{padding:24px}table{font-size:12px}.bar{min-width:120px}}
    """

    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine-tuning Strategy 최종 분석</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>Fine-tuning Strategy 최종 분석</h1>
    <p>Full Fine-tuning과 PEFT 전략을 <b>성능·효율·안정성</b> 3가지 지표로 비교한 최종 통합 분석본입니다.</p>
    <p class="small">생성 시각: {now.strftime('%Y-%m-%d %H:%M:%S')} KST · 전체 550개 run 완료 결과 기준</p>
    <div class="grid">
      <div class="stat"><b>{total_done}/550</b><span>전체 완료 run</span></div>
      <div class="stat"><b>{failed}</b><span>실패 run</span></div>
      <div class="stat"><b>{total_train_hours:.1f}h</b><span>순수 학습시간 합계</span></div>
      <div class="stat"><b>{epoch_examples:,}</b><span>Epoch 반영 학습 샘플</span></div>
    </div>
  </section>

  <h2>1. 연구 제목과 요약</h2>
  <section class="card paperbox">
    <h3>최고의 파인튜닝은 없다: 도메인을 가로지르는 성능·효율·안정성의 트레이드오프</h3>
    <p class="meta">There Is No Best Fine-Tuning Strategy: Performance, Efficiency, and Stability Trade-offs Across Domains</p>
    <p>
      본 연구는 Full Fine-tuning, LoRA, Adapter, IA³, BitFit을 동일한 분류 환경에서 비교하고, 각 방법을 단일 F1 점수가 아니라
      <b>성능(Performance), 효율(Efficiency), 안정성(Stability)</b>의 3가지 기준으로 평가한다.
      실험은 3개 Study, 22개 task/model 조합, 5개 방법, 5개 seed로 구성되며 총 550개 run을 포함한다.
    </p>
    <div class="note">
      <b>최종 결론:</b> Full Fine-tuning은 가장 강한 성능 기준선이지만 항상 최적은 아니다. Adapter와 LoRA는 훨씬 적은 학습 파라미터로 경쟁력 있는 성능을 보이며, IA³는 일부 hate-speech task에서 Full FT보다 높은 일반화 성능을 보였다. 또한 parameter-efficient 방법이 반드시 time-efficient하지는 않았다.
    </div>
  </section>

  <h2>연구 질문과 실험을 수행한 이유</h2>
  <section class="grid2">
    <div class="card">
      <h3>문제의식</h3>
      <p>
        사전학습 언어모델을 실제 분류 문제에 적용할 때 가장 흔한 선택지는 Full Fine-tuning이다.
        하지만 Full FT는 모든 가중치를 업데이트하므로 저장공간, 학습시간, 재현 비용이 크다.
        반대로 LoRA, Adapter, IA³, BitFit 같은 PEFT 방법은 일부 파라미터만 학습해 비용을 줄이지만,
        모든 데이터셋에서 같은 수준의 성능을 보장하는지는 별도로 확인해야 한다.
      </p>
      <p>
        따라서 이 실험의 핵심 질문은 “가장 높은 F1을 내는 방법이 항상 가장 좋은 방법인가?”가 아니라,
        <b>도메인, 언어, 데이터 크기, 모델 종류가 달라질 때 성능·시간·파라미터·안정성의 균형이 어떻게 바뀌는가</b>이다.
      </p>
    </div>
    <div class="card">
      <h3>왜 5개 방법을 비교했는가</h3>
      <p>
        Full FT는 성능 기준선 역할을 한다. LoRA와 Adapter는 실제 연구와 산업 적용에서 널리 쓰이는 대표적 PEFT 방법이다.
        IA³와 BitFit은 더 적은 파라미터만 조정하는 극단적으로 가벼운 전략에 가깝다.
        즉 5개 방법을 함께 비교하면 “성능을 위해 전체 모델을 학습해야 하는가”부터
        “아주 적은 파라미터만 학습해도 충분한가”까지 하나의 연속선으로 판단할 수 있다.
      </p>
      <p>
        단순히 한 데이터셋에서 한 번 돌린 결과가 아니라, 5개 seed 평균과 표준편차를 함께 보았기 때문에
        우연히 잘 나온 단일 run이 아니라 반복 실행에서 유지되는 경향을 판단할 수 있다.
      </p>
    </div>
  </section>

  <h2>실험이 실제로 비교한 것</h2>
  <section class="card">
    <p>
      본 실험은 “모델 하나를 특정 데이터셋에 맞게 다시 학습시키는 방법”을 비교한다.
      입력 문장은 tokenizer를 거쳐 모델에 들어가고, 모델은 각 문장이 어떤 라벨에 속하는지 예측한다.
      모든 방법은 동일한 train/validation/test split, 동일한 seed 목록, 동일한 최대 길이와 batch 조건에서 실행되었다.
      차이는 <b>어느 파라미터를 학습시키는가</b>에 있다.
    </p>
    <div class="flow">
      <div class="step"><b>데이터</b><span>뉴스, 리뷰, 트윗, 혐오표현 등 분류 데이터셋</span></div>
      <div class="step"><b>베이스 모델</b><span>BERTweet, RoBERTa-base, KLUE RoBERTa-base</span></div>
      <div class="step"><b>파인튜닝 전략</b><span>Full FT, LoRA, Adapter, IA³, BitFit</span></div>
      <div class="step"><b>반복 실행</b><span>각 조건당 5개 seed로 평균과 표준편차 계산</span></div>
      <div class="step"><b>평가</b><span>성능, 시간 효율, 파라미터 효율, 안정성 분리 평가</span></div>
    </div>
    <div class="note">
      <b>읽는 방법:</b> Macro-F1이 높으면 클래스별 균형 성능이 좋다는 뜻이고, F1 표준편차가 낮으면 seed가 바뀌어도 결과가 안정적이라는 뜻이다.
      학습시간이 짧으면 wall-clock 효율이 좋고, trainable parameter ratio가 낮으면 저장·배포 관점의 파라미터 효율이 좋다.
      이 네 기준은 서로 같은 방향으로 움직이지 않기 때문에 분리해서 해석해야 한다.
    </div>
  </section>

  <h2>Study별 설계 의도</h2>
  <section class="grid3">
    <div class="card">
      <h3>Study 1: 단일 도메인 심층 비교</h3>
      <p>
        BERTweet과 measuring_hate_speech 데이터셋을 사용해 트윗/소셜미디어 계열 혐오표현 분류에서
        5개 방법의 기본 성능 차이를 먼저 확인했다.
        이 Study는 전체 실험의 기준점으로, 동일 모델·동일 데이터셋에서 방법만 바꿨을 때 어떤 차이가 나는지 보여준다.
      </p>
    </div>
    <div class="card">
      <h3>Study 2: 영어 다중 도메인 확장</h3>
      <p>
        BERTweet과 FacebookAI/roberta-base를 9개 영어 task에 적용했다.
        트윗 특화 모델이 트윗 task에서 항상 유리한지, 일반 영어 모델이 비트윗 데이터에서 더 나은지,
        그리고 같은 데이터셋에서도 파인튜닝 방법에 따라 모델 간 우위가 바뀌는지를 보기 위한 설계다.
      </p>
    </div>
    <div class="card">
      <h3>Study 3: 한국어 데이터 검증</h3>
      <p>
        KLUE RoBERTa-base를 한국어 뉴스, 영화리뷰, 혐오표현 데이터셋에 적용했다.
        영어 중심 결과가 한국어에서도 유지되는지, 한국어 분류 문제에서는 Adapter나 LoRA가 Full FT를 대체할 수 있는지 확인했다.
      </p>
    </div>
  </section>

  <h2>2. 핵심 결과</h2>
  <section class="grid2">
    <div class="card"><h3>결과 1. 단일 최고 방법은 없었다</h3><p>Full FT가 강한 기준선이지만, Adapter와 IA³가 일부 조건에서 Full FT를 넘는 사례가 확인됐다.</p></div>
    <div class="card"><h3>결과 2. 3지표가 모두 필요했다</h3><p>성능 1위, 효율 1위, 안정성 1위가 같은 방법으로 수렴하지 않았다.</p></div>
    <div class="card"><h3>결과 3. 모델과 전략의 상호작용이 컸다</h3><p>BERTweet이 모든 트윗 task에서 항상 우세하지 않았고, RoBERTa-base와 특정 PEFT 조합이 더 강한 경우가 있었다.</p></div>
    <div class="card"><h3>결과 4. 파라미터 효율과 시간 효율은 달랐다</h3><p>LoRA는 학습 파라미터를 크게 줄였지만, 실제 wall-clock time까지 항상 줄이지는 않았다.</p></div>
  </section>

  <h2>3. 실험 설계</h2>
  <section class="card">
    <div class="flow">
      <div class="step"><b>3 Studies</b><span>영어 트윗 · 영어 다중 도메인 · 한국어 데이터</span></div>
      <div class="step"><b>5 Methods</b><span>Full FT, LoRA, Adapter, IA³, BitFit</span></div>
      <div class="step"><b>5 Seeds</b><span>42, 52, 62, 72, 82</span></div>
      <div class="step"><b>550 Runs</b><span>22 task/model × 5 methods × 5 seeds</span></div>
      <div class="step"><b>3 Metrics</b><span>성능 · 효율 · 안정성</span></div>
    </div>
  </section>
  <section class="card">{study_overview}</section>

  <h2>4. 실험 환경과 공통 설정</h2>
  <section class="grid2">
    <div class="card">
      <h3>로컬 실행 환경</h3>
      {table(["항목", "값"], [
        ["GPU", esc(env.get("gpu", "NVIDIA GeForce RTX 5070 Ti"))],
        ["GPU Memory", f"{esc(env.get('gpu_memory_gb', '15.92'))} GB"],
        ["CUDA Runtime", esc(env.get("cuda_runtime", "12.8"))],
        ["PyTorch", esc(env.get("torch", ""))],
        ["Transformers", esc(env.get("transformers", ""))],
        ["PEFT", esc(env.get("peft", ""))],
      ])}
    </div>
    <div class="card">
      <h3>공통 학습 설정</h3>
      {table(["항목", "값"], [
        ["Seeds", ", ".join(map(str, config.get("seeds", [42, 52, 62, 72, 82])))],
        ["Max length", str(config.get("max_length", 128))],
        ["Batch size", str(config.get("batch_size", 16))],
        ["Gradient accumulation", str(config.get("gradient_accumulation_steps", 4))],
        ["Effective batch", str(config.get("batch_size", 16) * config.get("gradient_accumulation_steps", 4))],
        ["Precision", esc(config.get("precision", "fp16"))],
        ["Weight decay", str(config.get("weight_decay", 0.01))],
      ])}
    </div>
  </section>

  <h2>5. 데이터셋과 실행 규모</h2>
  <section class="grid3">
    <div class="card"><h3>원본 학습 row 누적</h3><p>{raw_train_rows:,} rows</p><p class="meta">각 run의 train split 크기 합계</p></div>
    <div class="card"><h3>Epoch 반영 학습량</h3><p>{epoch_examples:,} examples</p><p class="meta">train rows × epochs 기준</p></div>
    <div class="card"><h3>Test 평가 row 누적</h3><p>{test_rows:,} rows</p><p class="meta">각 run의 test split 평가 누적</p></div>
  </section>
  <section class="card">{dataset_table}</section>

  <h2>6. 파인튜닝 방법 소개</h2>
  <section class="card">
    {method_intro_table}
  </section>

  <h2>7. 평가 프레임워크: 성능·효율·안정성</h2>
  <section class="grid3">
    <div class="card"><h3>성능 Performance</h3><span class="pill">Test Macro-F1</span><span class="pill">Accuracy</span><p>주 지표는 Macro-F1이다. 클래스 불균형이 있는 감성/혐오표현/뉴스 분류에서 Accuracy보다 더 균형 잡힌 판단을 제공한다.</p></div>
    <div class="card"><h3>효율 Efficiency</h3><span class="pill">Time Efficiency</span><span class="pill">Parameter Efficiency</span><p>효율은 하나의 값으로 묶지 않고 <b>시간 효율</b>과 <b>파라미터 효율</b>로 분리한다. 시간 효율은 평균 학습시간, 파라미터 효율은 trainable parameter ratio로 판단한다.</p></div>
    <div class="card"><h3>안정성 Stability</h3><span class="pill">5 Seeds</span><span class="pill">F1 SD</span><p>5개 seed 평균과 표준편차를 함께 제시해 재현성과 민감도를 확인한다. 낮은 표준편차는 안정성을 의미하지만 높은 성능을 보장하지는 않는다.</p></div>
  </section>

  <h2>8. 통합 결과 대시보드</h2>
  <section class="grid">
    <div class="card"><h3>성능 1위 분포</h3><p>{esc(count_text(performance_wins))}</p><p class="meta">각 task/model에서 평균 Macro-F1 최고 방법</p></div>
    <div class="card"><h3>시간 효율 1위 분포</h3><p>{esc(count_text(time_efficiency_wins))}</p><p class="meta">평균 학습시간이 가장 짧은 방법</p></div>
    <div class="card"><h3>파라미터 효율 1위 분포</h3><p>{esc(count_text(parameter_efficiency_wins))}</p><p class="meta">학습 파라미터 비율이 가장 낮은 방법</p></div>
    <div class="card"><h3>안정성 1위 분포</h3><p>{esc(count_text(stability_wins))}</p><p class="meta">Macro-F1 표준편차가 가장 낮은 방법</p></div>
  </section>
  <section class="card">
    <h3>방법별 전체 평균 경향</h3>
    {method_agg_table}
    <div class="note">
      <b>해석:</b> Full FT는 평균 성능과 순위에서 강하지만, 효율성은 낮다. 시간 효율은 BitFit 또는 Adapter가 우세한 경우가 많고, 파라미터 효율은 IA³가 가장 강하다. LoRA는 파라미터 효율적이지만 wall-clock time이 항상 짧지는 않아 시간 효율과 파라미터 효율을 분리해서 판단해야 한다.
    </div>
  </section>

  <h2>9. 지표별 1위 방법</h2>
  <section class="card">
    <p class="meta">각 Study·데이터셋·모델 조합에서 성능, 시간 효율, 파라미터 효율, 안정성 기준으로 가장 좋은 방법을 분리해 표시했다.</p>
    {indicator_winner_table}
    <div class="note">
      <b>해석:</b> 같은 데이터셋에서도 성능 1위와 효율 1위가 다르게 나타난다. 따라서 “가장 좋은 방법”은 단일 순위가 아니라, 어떤 기준을 우선하느냐에 따라 달라진다.
    </div>
  </section>

  <h2>10. Study별 평균 결과</h2>
  <section class="card">
    {study_agg_table}
    <div class="note">
      <b>해석:</b> Study 1/2에서는 Full FT의 성능 우위가 강하게 나타나지만, Study 3의 YNAT처럼 Adapter가 Full FT를 넘는 사례도 존재한다. 평균값만 보면 Full FT가 강하지만, task별 결과를 보면 PEFT의 일반화 이점과 효율 이점이 분리되어 나타난다.
    </div>
  </section>

  <h2>11. Study별 핵심 인사이트</h2>
  <section class="grid3">
    <div class="card"><h3>Study 1</h3><p>BERTweet 기반 hate-speech 실험에서는 Full FT가 최고 성능을 보였고, Adapter는 작은 성능 손실로 시간과 파라미터 효율을 확보했다.</p></div>
    <div class="card"><h3>Study 2</h3><p>영어 다중 도메인에서는 Full FT 우위가 강하지만, tweet_hate에서 IA³가 Full FT보다 높은 성능을 보이는 예외가 나타났다.</p></div>
    <div class="card"><h3>Study 3</h3><p>한국어 YNAT에서는 Adapter가 Full FT보다 높은 성능을 보였고, NSMC/K-MHaS에서는 Full FT가 강했다. 즉 언어와 데이터셋에 따라 전략 우위가 달라진다.</p></div>
  </section>

  <h2>12. 주요 인사이트</h2>
  <section class="grid2">
    <div class="card"><h3>Insight 1. Full FT는 강하지만 절대적 기준은 아니다</h3><p>대부분의 task에서 Full FT가 높은 성능을 보였지만, Adapter와 IA³가 Full FT를 넘는 사례가 존재했다. 따라서 Full FT는 기준선이지 항상 최종 선택지는 아니다.</p></div>
    <div class="card"><h3>Insight 2. 성능 1위와 효율 1위는 자주 다르다</h3><p>성능은 Full FT가 강한 반면, 시간 효율은 BitFit/Adapter, 파라미터 효율은 IA³가 강하게 나타났다. 하나의 점수로 방법을 고르면 실제 선택 기준을 놓칠 수 있다.</p></div>
    <div class="card"><h3>Insight 3. 효율은 반드시 두 개로 나눠야 한다</h3><p>파라미터 효율이 높은 방법이 항상 빠르지는 않았다. LoRA는 파라미터 수를 크게 줄였지만 일부 조건에서 Full FT보다 빠르지 않았다.</p></div>
    <div class="card"><h3>Insight 4. Adapter는 균형형 선택지다</h3><p>Adapter는 Full FT를 항상 이기지는 않지만, 여러 데이터셋에서 성능 손실을 작게 유지하면서 시간과 파라미터를 줄였다. 성능과 효율을 동시에 고려할 때 가장 현실적인 후보 중 하나다.</p></div>
    <div class="card"><h3>Insight 5. IA³는 평균보다 예외가 중요하다</h3><p>IA³는 평균 성능만 보면 강한 방법은 아니지만, tweet_hate처럼 특정 조건에서 Full FT보다 높은 성능을 보였다. 작은 데이터셋이나 불안정한 분포에서 제한적 업데이트가 일반화에 유리할 수 있다.</p></div>
    <div class="card"><h3>Insight 6. BitFit은 초경량 하한선이다</h3><p>BitFit은 빠르고 단순하지만 복잡한 task에서는 성능 손실이 커졌다. 따라서 최종 선택지라기보다 가장 가벼운 기준선으로 해석하는 것이 적절하다.</p></div>
    <div class="card"><h3>Insight 7. 안정성은 성능과 다르다</h3><p>표준편차가 낮은 방법이 항상 가장 높은 평균 성능을 보이지는 않았다. 안정성은 재현성의 지표이고, 성능 우위와 별도로 판단해야 한다.</p></div>
    <div class="card"><h3>Insight 8. 도메인 특화 모델은 충분조건이 아니다</h3><p>BERTweet은 트윗 중심 모델이지만 모든 트윗 task에서 항상 우세하지 않았다. 모델의 사전학습 도메인보다 데이터셋 특성과 fine-tuning strategy의 조합이 더 중요할 수 있다.</p></div>
    <div class="card"><h3>Insight 9. RoBERTa-base도 트윗 task에서 경쟁력이 있다</h3><p>RoBERTa-base는 범용 모델이지만 특정 PEFT 전략과 결합했을 때 트윗 task에서 BERTweet 조합을 넘는 사례가 있었다.</p></div>
    <div class="card"><h3>Insight 10. 한국어 task에서도 전략 우위는 고정되지 않았다</h3><p>YNAT에서는 Adapter가 강했고, NSMC와 K-MHaS에서는 Full FT가 강했다. 같은 한국어 모델을 써도 데이터셋 성격에 따라 최적 전략이 달라졌다.</p></div>
    <div class="card"><h3>Insight 11. 데이터셋 크기가 trade-off를 바꾼다</h3><p>큰 데이터셋에서는 Full FT의 표현 적응력이 강하게 작동하는 경향이 있었고, 작은 데이터셋에서는 PEFT의 제한적 업데이트가 과적합을 줄이는 방향으로 작동할 수 있었다.</p></div>
    <div class="card"><h3>Insight 12. 선택 문제는 순위 문제가 아니라 조건 문제다</h3><p>최종 선택은 “어떤 방법이 1위인가”보다 “성능 손실을 얼마나 허용하고 시간/파라미터를 얼마나 줄일 것인가”의 문제로 보는 것이 더 정확하다.</p></div>
  </section>

  <h2>13. 주목할 만한 수치 사례</h2>
  <section class="card">
    {notable_table}
  </section>

  <h2>14. 도메인 특화 모델은 항상 유리한가?</h2>
  <section class="card">
    <p>BERTweet은 트윗 중심 사전학습 모델이지만 모든 트윗 task에서 항상 우세하지는 않았다. RoBERTa-base에 특정 PEFT 전략을 적용했을 때 BERTweet 조합보다 더 높은 성능을 보이는 사례가 존재했다.</p>
    {interaction_table}
    <div class="note warn">
      <b>해석:</b> 사전학습 도메인 적합성은 중요하지만 충분조건은 아니다. 최종 성능은 모델, 데이터셋, 파인튜닝 전략의 상호작용으로 결정된다.
    </div>
  </section>

  <h2>15. 전체 세부 결과</h2>
  <p class="meta">아래 표는 22개 task/model 조합 각각에 대해 5개 방법의 평균 성능, 표준편차, 시간, 파라미터 효율을 비교한다.</p>
  {''.join(cards)}

  <h2>16. 전략 선택 가이드</h2>
  <section class="card">
    {table(["상황", "우선 고려할 방법", "근거"], [
        ["최고 성능이 최우선인 경우", "Full Fine-tuning", "대부분의 task/model 조합에서 가장 강한 성능 기준선으로 작동했다."],
        ["성능 손실을 작게 유지하면서 비용을 줄이고 싶은 경우", "Adapter", "여러 데이터셋에서 Full FT에 근접하고 시간 절감도 비교적 안정적으로 나타났다."],
        ["저장 공간과 학습 파라미터를 크게 줄이고 싶은 경우", "LoRA", "파라미터 비율은 낮지만 실제 시간 효율은 task별로 확인해야 한다."],
        ["아주 작은 학습 파라미터로 빠르게 시도해야 하는 경우", "IA³ 또는 BitFit", "평균 성능은 낮을 수 있지만 초경량 기준선으로 유용하다."],
        ["작거나 불안정한 hate-speech 데이터인 경우", "IA³도 함께 검토", "tweet_hate에서 Full FT보다 높은 일반화 성능을 보인 사례가 있었다."],
        ["트윗 데이터라고 해서 모델을 고정하려는 경우", "BERTweet과 RoBERTa 모두 비교", "도메인 특화 모델의 우위가 fine-tuning strategy에 따라 뒤집힐 수 있었다."],
    ])}
  </section>

  <h2>17. 종합 결론</h2>
  <section class="card paperbox">
    <p>
      본 연구는 동일한 모델과 데이터셋 조건에서도 파인튜닝 전략에 따라 <b>성능, 효율, 안정성</b>이 크게 달라짐을 보였다.
      Full Fine-tuning은 전반적으로 강한 성능 기준선으로 작동했지만, Adapter와 LoRA는 훨씬 적은 학습 파라미터로 경쟁력 있는 성능을 유지했으며,
      IA³는 일부 hate-speech task에서 Full Fine-tuning보다 더 나은 일반화 성능을 보였다.
      또한 도메인 특화 모델이 항상 최적의 선택은 아니며, parameter-efficient 방법이 반드시 time-efficient하지도 않다는 점을 확인했다.
      따라서 실제 응용 환경에서는 단일 F1 점수만을 기준으로 파인튜닝 방법을 선택하기보다, 성능·효율·안정성의 3가지 판단 기준을 함께 고려해야 한다.
      특히 효율은 시간 효율과 파라미터 효율로 분리해 보아야 하며, 두 효율 지표가 항상 같은 방향으로 움직이지 않는다는 점이 확인되었다.
    </p>
  </section>

  <h2>18. 한계와 추가 검증</h2>
  <section class="grid2">
    <div class="card"><h3>한계</h3><ul><li>모든 실험은 단일 로컬 GPU 환경에서 수행되어 하드웨어 일반화에는 주의가 필요하다.</li><li>학습시간은 구현체와 라이브러리 버전에 영향을 받을 수 있다.</li><li>분류 task 중심이므로 생성 task나 대형 LLM으로 일반화하려면 추가 실험이 필요하다.</li></ul></div>
    <div class="card"><h3>추가 검증</h3><ul><li>paired t-test 또는 Wilcoxon test로 seed별 차이의 유의성 검정</li><li>성능-효율 Pareto frontier 시각화</li><li>데이터셋 크기와 PEFT 성능 손실 간 상관 분석</li><li>결과 표를 평균±표준편차 형식으로 정리</li></ul></div>
  </section>
</main>
</body>
</html>
"""

    OUT.write_text(doc, encoding="utf-8-sig")
    print(OUT.resolve(), OUT.stat().st_size)


if __name__ == "__main__":
    make_report()
