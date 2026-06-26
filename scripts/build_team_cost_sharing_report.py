from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "TEAM_COST_SHARING_REPORT.html"


def esc(x) -> str:
    return html.escape(str(x))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def load_unique_runs():
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
    return [item for _, item in latest.values()]


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    rows = load_unique_runs()
    progress = {
        study: read_json(RESULTS / study / "PAPER" / "progress.json") or {}
        for study in ["study1", "study2", "study3"]
    }
    total_runs = len(rows)
    failed = sum(int(progress[s].get("failed", 0) or 0) for s in progress)
    train_hours = sum(float(x.get("train_seconds", 0) or 0) for x in rows) / 3600
    epoch_examples = sum(
        int(x.get("train_rows", 0) or 0) * int(x.get("epochs_requested", 0) or 0)
        for x in rows
    )
    train_rows_sum = sum(int(x.get("train_rows", 0) or 0) for x in rows)
    test_rows_sum = sum(int(x.get("test_rows", 0) or 0) for x in rows)

    # Prior runtime observation from the experiment monitoring.
    wall_hours_low = 50
    wall_hours_high = 52
    realistic_power_kw = 0.40
    high_power_kw = 0.55
    realistic_kwh_low = wall_hours_low * realistic_power_kw
    realistic_kwh_high = wall_hours_high * realistic_power_kw
    high_kwh = wall_hours_high * high_power_kw
    won_per_kwh_low = 200
    won_per_kwh_high = 300
    elec_low = realistic_kwh_low * won_per_kwh_low
    elec_high = high_kwh * won_per_kwh_high

    # Suggested contribution: electricity + PC occupation + monitoring/operation token.
    suggested_per_person = 30000
    friendly_per_person = 20000
    hardline_per_person = 30000
    now = dt.datetime.now()

    css = """
    :root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687386;--line:#e4e8f0;--blue:#2563eb;--green:#059669;--red:#b42318;--amber:#d97706}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.58}
    .wrap{max-width:1080px;margin:0 auto;padding:34px 22px 80px}.hero{background:linear-gradient(135deg,#111827,#2563eb);color:white;border-radius:28px;padding:36px;box-shadow:0 20px 52px rgba(25,40,80,.22)}
    h1{margin:0 0 10px;font-size:34px;letter-spacing:-.04em} h2{margin:34px 0 14px;font-size:23px;letter-spacing:-.03em} h3{margin:0 0 8px;font-size:18px;letter-spacing:-.02em}.hero p{margin:7px 0;color:#e9efff}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:16px}.stat b{display:block;font-size:27px}.stat span{font-size:13px;color:#dbe5ff}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin:14px 0;box-shadow:0 8px 24px rgba(20,30,55,.05)}.meta{color:var(--muted);font-size:13px;margin:4px 0 14px}
    .note{background:#f8fafc;border-left:4px solid var(--blue);padding:14px 16px;border-radius:12px;color:#2a3345;margin:12px 0}.money{border-left-color:var(--green);background:#f4fbf7}.warn{border-left-color:var(--amber)}
    table{width:100%;border-collapse:collapse;font-size:14px} th,td{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle} th{background:#f1f4fa;color:#465164;font-weight:800}
    .price{font-size:30px;font-weight:900;color:var(--green)}.small{font-size:13px;color:#dbe5ff}.tag{display:inline-block;background:#eef4ff;color:#2457d6;border:1px solid #d7e4ff;border-radius:999px;padding:5px 10px;font-size:12px;font-weight:800;margin:4px 6px 4px 0}
    ul{margin:8px 0 0 18px;padding:0} li{margin:7px 0}.center{text-align:center}
    @media(max-width:900px){.grid,.grid2,.grid3{grid-template-columns:1fr}.wrap{padding:20px 12px}.hero{padding:24px}table{font-size:12px}}
    """

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>팀 실험 인프라 정산 보고서</title>
  <style>{css}</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>팀 실험 인프라 정산 보고서</h1>
    <p>550회 로컬 GPU 학습 실험을 완료하기 위해 사용된 개인 PC, 전력, 장비 점유, 실험 운영 시간을 정리한 팀 비용 분담 보고서입니다.</p>
    <p class="small">생성 시각: {now.strftime('%Y-%m-%d %H:%M:%S')} KST · 대상: 홍유석, 이수용</p>
    <div class="grid">
      <div class="stat"><b>{total_runs}/550</b><span>완료된 학습 run</span></div>
      <div class="stat"><b>{failed}</b><span>실패 run</span></div>
      <div class="stat"><b>{train_hours:.1f}h</b><span>순수 학습시간 합계</span></div>
      <div class="stat"><b>{epoch_examples:,}</b><span>epoch 반영 학습 샘플</span></div>
    </div>
  </section>

  <h2>1. 정산 요약</h2>
  <section class="grid3">
    <div class="card center">
      <h3>전기세만 계산</h3>
      <p class="price">3,000원</p>
      <p class="meta">순수 전력비만 나눈 최소 금액. 실제 기여와 장비 점유는 반영되지 않음.</p>
    </div>
    <div class="card center">
      <h3>기본 분담안</h3>
      <p class="price">20,000원</p>
      <p class="meta">전기세, PC 점유, 장시간 모니터링을 최소한으로 반영한 금액.</p>
    </div>
    <div class="card center">
      <h3>권장 분담안</h3>
      <p class="price">30,000원</p>
      <p class="meta">코드 작성, 실험 운영, 개인 장비 제공, 결과 정리까지 포함한 현실적 분담 기준.</p>
    </div>
  </section>
  <section class="card">
    <div class="note money">
      <b>정산 기준:</b> 인당 30,000원이 가장 합리적입니다.
      전기세만 보면 금액이 작지만, 실제로는 개인 PC를 약 이틀간 실험 장비로 점유했고, 실험 설계·코드 작성·모니터링·결과 정리·보고서 제작까지 포함된 운영 비용입니다.
    </div>
  </section>

  <h2>2. 실제 수행한 작업</h2>
  <section class="card">
    {table(["구분", "수행 내용", "비고"], [
        ["실험 설계", "Full FT, LoRA, Adapter, IA³, BitFit 5개 방법 비교 설계", "논문 주제 핵심"],
        ["코드 작성", "Jupyter Notebook, 실행 스크립트, 저장/재시작 구조 구성", "중간 중단 대응 가능"],
        ["실험 실행", "로컬 RTX 5070 Ti 환경에서 전체 550 run 수행", "개인 PC 장시간 점유"],
        ["모니터링", "진행률, GPU 상태, 실패 여부, 남은 시간 반복 확인", "실패 0개로 완주"],
        ["결과 정리", "최종 HTML 보고서, CSV 표, 논문 인사이트 정리", "보고/논문 작성용"],
    ])}
  </section>

  <h2>3. 실험 규모</h2>
  <section class="grid3">
    <div class="card"><h3>학습 데이터 누적</h3><p class="price">{train_rows_sum:,}</p><p class="meta">각 run의 train rows 합계</p></div>
    <div class="card"><h3>Epoch 반영 학습량</h3><p class="price">{epoch_examples:,}</p><p class="meta">train rows × epochs 기준</p></div>
    <div class="card"><h3>평가 데이터 누적</h3><p class="price">{test_rows_sum:,}</p><p class="meta">각 run의 test rows 합계</p></div>
  </section>
  <section class="card">
    {table(["Study", "Run 수", "상태"], [
        ["Study 1", "25/25", "완료"],
        ["Study 2", "450/450", "완료"],
        ["Study 3", "75/75", "완료"],
        ["전체", "550/550", "실패 0개"],
    ])}
  </section>

  <h2>4. 전력 사용량과 실제 비용 추정</h2>
  <section class="card">
    {table(["가정", "계산", "예상 비용"], [
        ["낮은 추정", f"약 {realistic_kwh_low:.1f} kWh × 200원/kWh", f"약 {elec_low:,.0f}원"],
        ["현실적 추정", f"약 {realistic_kwh_high:.1f} kWh × 250원/kWh", f"약 {realistic_kwh_high * 250:,.0f}원"],
        ["높은 추정", f"약 {high_kwh:.1f} kWh × 300원/kWh", f"약 {elec_high:,.0f}원"],
    ])}
    <div class="note warn">
      <b>주의:</b> 전기요금은 누진 구간, 기본요금, 기후환경요금에 따라 달라질 수 있습니다. 위 계산은 실험으로 인한 추가 전력 사용량을 대략 추정한 값입니다.
    </div>
  </section>

  <h2>5. 인당 30,000원이 합리적인 이유</h2>
  <section class="card">
    {table(["항목", "근거", "가치 산정"], [
        ["전력비", "약 50시간 이상 로컬 GPU 학습 가동", "약 5,000~8,000원 수준"],
        ["PC/GPU 점유", "RTX 5070 Ti 장비를 실험 기간 동안 사실상 전용 학습 장비로 사용", "개인 장비 사용료 성격"],
        ["장비 부하", "GPU, CPU, SSD, 팬, 파워서플라이에 장시간 지속 부하", "소모성·위험 부담 포함"],
        ["실험 운영", "550개 run 진행률 확인, 오류 여부 확인, 남은 시간 계산", "단순 전기세보다 큰 운영 노동"],
        ["코드/자동화", "Jupyter notebook, 저장 구조, 재시작 가능 구조, 결과 집계 코드 구성", "팀 공동 산출물 생성"],
        ["보고서 제작", "교수님/논문용 HTML, CSV 표, 최종 분석 리포트 생성", "제출물 품질 향상"],
    ])}
    <div class="note money">
      <b>계산 논리:</b> 전기세는 실제 비용의 일부에 불과합니다. 개인 GPU PC를 장시간 제공하고, 실험이 끝날 때까지 운영·검증·결과 정리까지 수행한 점을 고려하면 인당 30,000원은 GPU 서버/Colab Pro/외부 대여 비용보다 훨씬 낮은 내부 분담 기준입니다.
    </div>
  </section>

  <h2>6. 외부 대안 대비 비용 비교</h2>
  <section class="card">
    {table(["대안", "예상 비용/문제", "이번 방식의 장점"], [
        ["Colab 무료/저가 환경", "세션 끊김, Drive 용량 제한, 장시간 550 run 안정성 불확실", "로컬에서 중단 없이 550/550 완료"],
        ["Colab Pro/Pro+ 또는 클라우드 GPU", "월 구독료 또는 시간당 GPU 비용 발생", "추가 결제 없이 개인 장비로 완료"],
        ["외부 GPU 서버 대여", "GPU 시간당 비용 + 데이터 이동 + 환경 세팅 필요", "이미 세팅된 로컬 환경에서 즉시 실행"],
        ["팀원이 각자 나눠 실행", "환경 차이로 논문 실험 일관성 훼손 가능", "동일 PC/동일 환경에서 결과 신뢰도 확보"],
    ])}
  </section>

  <h2>7. 왜 전기세만 보면 안 되는가</h2>
  <section class="grid2">
    <div class="card">
      <h3>PC 점유 비용</h3>
      <ul>
        <li>개인 PC를 장시간 학습 장비로 사용</li>
        <li>GPU, CPU, SSD, 팬, 파워서플라이에 지속 부하 발생</li>
        <li>실험 중 게임/작업/재부팅/업데이트 제한</li>
      </ul>
    </div>
    <div class="card">
      <h3>운영 노동</h3>
      <ul>
        <li>실험 중단 여부 확인</li>
        <li>남은 시간 계산</li>
        <li>결과 파일 정상 저장 여부 확인</li>
        <li>교수님/논문용 보고서 생성</li>
      </ul>
    </div>
  </section>

  <h2>8. 팀원에게 보낼 문구</h2>
  <section class="card paperbox">
    <p>
      이번 논문 실험은 한 PC에서 550개 run을 전부 돌렸고, 실험 기간 동안 개인 GPU PC를 계속 학습 장비로 사용했습니다.
      전기세만 보면 크지 않지만, 실제로는 실험 세팅, 코드 작성, 실행 관리, 모니터링, 결과 정리, 최종 보고서 제작까지 포함된 작업이었습니다.
      외부 GPU 서버나 Colab Pro 환경을 사용했을 때보다 훨씬 낮은 비용으로 안정적으로 끝낸 것이므로, 장비 사용과 운영 수고를 포함한 팀 내부 분담금으로 <b>인당 30,000원</b>이 합리적입니다.
    </p>
  </section>

  <h2>9. 최종 정산표</h2>
  <section class="card">
    {table(["이름", "역할", "요청 금액", "비고"], [
        ["홍유석", "팀원", f"{suggested_per_person:,}원", "실험 인프라 분담금"],
        ["이수용", "팀원", f"{suggested_per_person:,}원", "실험 인프라 분담금"],
        ["본인", "코드 작성·실험 실행·PC 제공·모니터링·결과 정리", "0원", "노동과 장비 제공으로 기여"],
    ])}
    <div class="note money">
      <b>권장 총 정산액:</b> 60,000원. 이 금액은 단순 전기세가 아니라 개인 장비 점유, 장시간 실험 운영, 코드 작성, 결과 보고서 제작까지 포함한 팀 내부 분담 기준입니다.
    </div>
  </section>
</main>
</body>
</html>
"""

    OUT.write_text(html_doc, encoding="utf-8-sig")
    print(OUT.resolve(), OUT.stat().st_size)


if __name__ == "__main__":
    main()
