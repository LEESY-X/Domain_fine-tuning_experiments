from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    candidates = [
        {
            "rank": "1순위",
            "title_ko": "최고의 파인튜닝은 없다: 도메인을 가로지르는 성능·안정성·효율성의 트레이드오프",
            "title_en": "There Is No Best Fine-Tuning Strategy: Performance, Stability, and Efficiency Trade-offs Across Domains",
            "fit": "가장 논문 제목답고, 현재 실험 설계 전체를 가장 정확히 담습니다.",
            "conclusion": (
                "본 연구의 핵심 결론은 ‘최고의 파인튜닝 방법은 고정되어 있지 않다’는 것이다. "
                "Full Fine-tuning은 전반적으로 가장 강한 성능 기준선으로 작동했지만, Adapter와 LoRA는 훨씬 적은 학습 파라미터로 경쟁력 있는 성능을 유지했으며, "
                "IA³는 일부 hate-speech task에서 Full Fine-tuning보다 더 나은 일반화 성능을 보였다. "
                "이는 파인튜닝 전략의 우위가 단일 F1 점수나 파라미터 수가 아니라 데이터셋의 도메인, 크기, 분포 특성, 사전학습 모델과의 적합성에 의해 결정됨을 보여준다."
            ),
        },
        {
            "rank": "2순위",
            "title_ko": "F1의 함정: 파인튜닝 전략의 승자는 왜 데이터셋마다 달라지는가",
            "title_en": "The F1 Trap: Why the Winning Fine-Tuning Strategy Changes Across Datasets",
            "fit": "임팩트가 가장 강합니다. 발표나 포스터 제목으로 특히 좋습니다.",
            "conclusion": (
                "본 연구는 F1 점수만으로 파인튜닝 전략을 선택하는 방식이 실제 의사결정에 충분하지 않음을 보였다. "
                "Full Fine-tuning은 여러 데이터셋에서 최고 성능을 보였지만, Adapter는 작은 성능 손실로 학습시간과 파라미터 효율을 개선했고, "
                "LoRA는 특정 task에서 Full Fine-tuning에 근접한 성능을 보였으며, IA³는 일부 불안정한 hate-speech 데이터에서 오히려 더 높은 일반화 성능을 보였다. "
                "따라서 파인튜닝 방법의 평가는 성능뿐 아니라 안정성, 시간, 자원 효율을 함께 고려해야 한다."
            ),
        },
        {
            "rank": "3순위",
            "title_ko": "파인튜닝 전략 선택의 재고: Full Fine-tuning과 PEFT의 도메인별 비교",
            "title_en": "Rethinking Fine-Tuning Strategy Selection: A Domain-wise Comparison of Full Fine-Tuning and PEFT",
            "fit": "가장 안정적이고 교수님께 설명하기 좋습니다. 과장 없이 학술적입니다.",
            "conclusion": (
                "본 연구는 Full Fine-tuning, LoRA, Adapter, IA³, BitFit을 동일한 조건에서 비교하여, "
                "파인튜닝 전략 선택이 데이터셋과 모델의 조합에 따라 달라짐을 확인하였다. "
                "Full Fine-tuning은 강한 기준선으로 기능했으나, PEFT 방법들은 훨씬 적은 학습 파라미터로도 의미 있는 성능을 달성했다. "
                "특히 Adapter는 여러 task에서 성능과 효율의 균형이 우수했으며, LoRA와 IA³는 task 특성에 따라 상반된 결과를 보였다. "
                "이는 실제 응용에서 단일 방법을 고정적으로 선택하기보다 task 조건에 맞는 전략 선택이 필요함을 시사한다."
            ),
        },
        {
            "rank": "4순위",
            "title_ko": "파라미터 효율성은 시간 효율성을 보장하는가?: 다중 도메인 파인튜닝 전략 비교",
            "title_en": "Does Parameter Efficiency Guarantee Time Efficiency? A Multi-Domain Comparison of Fine-Tuning Strategies",
            "fit": "LoRA가 항상 빠르지 않았던 결과를 전면에 내세우는 제목입니다.",
            "conclusion": (
                "본 연구는 parameter-efficient fine-tuning 방법이 반드시 time-efficient하지는 않음을 보였다. "
                "LoRA와 Adapter는 학습 파라미터 수를 크게 줄였지만, 실제 학습시간은 task와 구현 오버헤드에 따라 Full Fine-tuning과 비슷하거나 더 길어지는 경우도 있었다. "
                "반면 Adapter는 다수의 데이터셋에서 비교적 안정적인 성능-시간 균형을 보였고, Full Fine-tuning은 여전히 강력한 성능 기준선이었다. "
                "따라서 파인튜닝 전략을 평가할 때는 파라미터 수만이 아니라 실제 wall-clock time과 성능 안정성을 함께 측정해야 한다."
            ),
        },
        {
            "rank": "5순위",
            "title_ko": "도메인 특화 모델은 항상 유리한가?: 사전학습 모델과 파인튜닝 전략의 상호작용 분석",
            "title_en": "Are Domain-Specific Models Always Better? An Analysis of Interactions Between Pretrained Models and Fine-Tuning Strategies",
            "fit": "BERTweet vs RoBERTa 비교를 강하게 밀고 싶을 때 좋습니다.",
            "conclusion": (
                "본 연구는 도메인 특화 사전학습 모델이 항상 우수한 결과를 보장하지 않음을 보였다. "
                "BERTweet은 일부 트윗 감성 task에서 강점을 보였지만, 모든 트윗 기반 데이터셋에서 범용 RoBERTa를 일관되게 앞서지는 않았다. "
                "또한 동일한 사전학습 모델에서도 Full Fine-tuning, Adapter, LoRA, IA³, BitFit의 상대적 성능은 task별로 달라졌다. "
                "이는 최종 성능이 사전학습 도메인뿐 아니라 fine-tuning strategy와 데이터셋 특성의 상호작용에 의해 결정됨을 시사한다."
            ),
        },
    ]

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2457d6;--green:#11845b;--amber:#9a6700}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.6}
    .wrap{max-width:1080px;margin:0 auto;padding:34px 22px 70px}
    .hero{background:linear-gradient(135deg,#172033,#2457d6);color:white;border-radius:24px;padding:34px;box-shadow:0 18px 45px rgba(25,40,80,.18)}
    h1{margin:0 0 10px;font-size:30px;letter-spacing:-.03em} h2{margin:30px 0 14px;font-size:22px;letter-spacing:-.02em} h3{margin:0 0 8px;font-size:20px;letter-spacing:-.02em}
    .hero p{margin:6px 0;color:#e9efff}.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:24px;margin:16px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}
    .rank{display:inline-block;background:#eaf1ff;color:var(--blue);font-weight:800;padding:4px 10px;border-radius:999px;margin-bottom:12px}
    .en{color:#3b4658;font-size:15px;margin:4px 0 14px}.fit{background:#f8fafc;border-left:4px solid var(--blue);padding:12px 14px;border-radius:12px;margin:12px 0;color:#2a3345}
    .conclusion{border-top:1px solid var(--line);margin-top:16px;padding-top:16px}.label{font-weight:800;color:#111827}.recommend{border:2px solid #9db7ff}
    ul{margin:10px 0 0 20px;padding:0} li{margin:6px 0}.small{font-size:13px;color:#dbe5ff}.pick{color:var(--green);font-weight:800}
    """

    cards = []
    for idx, c in enumerate(candidates):
        cls = "card recommend" if idx == 0 else "card"
        cards.append(
            f"""
            <section class="{cls}">
              <span class="rank">{c['rank']}</span>
              <h3>{c['title_ko']}</h3>
              <p class="en">{c['title_en']}</p>
              <div class="fit"><span class="label">추천 이유:</span> {c['fit']}</div>
              <div class="conclusion">
                <p><span class="label">결론 문장:</span></p>
                <p>{c['conclusion']}</p>
              </div>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>논문 제목 및 결론 후보</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>논문 제목 및 결론 후보</h1>
    <p>현재 실험 결과 흐름에 맞춰 제목과 결론 문장을 함께 묶은 후보군입니다.</p>
    <p class="small">가장 추천하는 방향은 1순위입니다. 임팩트 중심 발표용은 2순위가 적합합니다.</p>
  </section>

  <h2>추천 후보</h2>
  {''.join(cards)}

  <section class="card">
    <h3>최종 추천</h3>
    <p><span class="pick">논문 제출용으로는 1순위</span>를 추천합니다. 실험의 전체 범위인 Full FT와 PEFT 비교, 도메인별 차이, 성능·안정성·효율성 trade-off를 가장 균형 있게 담습니다.</p>
    <p>발표나 포스터처럼 임팩트를 더 강하게 주고 싶다면 2순위 제목을 사용할 수 있습니다.</p>
  </section>
</main>
</body>
</html>
"""

    out = ROOT / "TITLE_CONCLUSION_CANDIDATES.html"
    out.write_text(doc, encoding="utf-8-sig")
    print(out.resolve(), out.stat().st_size)


if __name__ == "__main__":
    main()
