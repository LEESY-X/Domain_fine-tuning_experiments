import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()], **({"execution_count": None, "outputs": []} if kind == "code" else {})}


def notebook(title, study=None, action=None):
    note = "기본 실행 모드는 전체 논문 실험용 `PAPER`입니다. 짧은 점검만 필요할 때 `SMOKE`로 변경하세요." if study else "로컬 실행 환경과 결과 파일을 확인합니다."
    cells = [cell("markdown", f"# {title}\n\n{note}")]
    cells.append(cell("code", "from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom src.suite import precheck, run_study, aggregate\nprecheck()"))
    if study:
        cells.append(cell("code", f"RUN_MODE = 'PAPER'  # 짧은 점검만 할 때: 'SMOKE'\nMAX_JOBS = 1 if RUN_MODE == 'SMOKE' else None\nSTUDY = '{study}'\nprint(f'{{STUDY=}}, {{RUN_MODE=}}, {{MAX_JOBS=}}')"))
        cells.append(cell("markdown", "## 실행\n\n각 run과 epoch가 즉시 디스크에 저장됩니다. 중단 후 이 셀을 다시 실행하면 완료된 run은 자동으로 건너뜁니다."))
        cells.append(cell("code", "result = run_study(STUDY, run_mode=RUN_MODE, max_jobs=MAX_JOBS)\ndisplay(result.tail())"))
        cells.append(cell("code", "progress = ROOT / 'results' / STUDY / RUN_MODE / 'progress.json'\nprint(progress.read_text(encoding='utf-8') if progress.exists() else '아직 진행 파일 없음')"))
    elif action == "aggregate":
        cells.append(cell("code", "RUN_MODE = 'PAPER'\nall_runs = aggregate(RUN_MODE)\ndisplay(all_runs.head())\nprint('runs:', len(all_runs))"))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (ai_lab_first)", "language": "python", "name": "ai_lab_first"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def monitor_notebook():
    cells = [
        cell("markdown", "# PAPER 실행 진행상태 모니터\n\n학습 노트북과 다른 커널에서 실행하면 학습 중에도 상태를 확인할 수 있습니다."),
        cell("code", "from pathlib import Path\nimport json\nfrom collections import Counter\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent"),
        cell("code", "STUDY = 'study1'  # study1, study2, study3\nRUN_MODE = 'PAPER'"),
        cell("code", "root = ROOT / 'results' / STUDY / RUN_MODE\nrows = []\nfor path in root.glob('**/status.json') if root.exists() else []:\n    try:\n        row = json.loads(path.read_text(encoding='utf-8')); row['path'] = str(path.parent.relative_to(ROOT)); rows.append(row)\n    except Exception:\n        rows.append({'status':'UNREADABLE','path':str(path.parent.relative_to(ROOT))})\nprint('STATUS', dict(Counter(x.get('status','UNKNOWN') for x in rows)))\nfor x in rows:\n    if x.get('status') in {'RUNNING','FAILED'}: print(x.get('status'),x.get('task'),x.get('model'),x.get('method'),x.get('seed'),x.get('resumed_from_checkpoint'))\nprogress = root / 'progress.json'\nprint(progress.read_text(encoding='utf-8') if progress.exists() else 'progress.json 없음')"),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (ai_lab_first)", "language": "python", "name": "ai_lab_first"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}


specs = {
    "00_PRECHECK.ipynb": ("Local 5-Method Suite Precheck", None, None),
    "01_STUDY1_BERTWEET_HATE.ipynb": ("Study 1 - BERTweet Hate Speech", "study1", None),
    "02_STUDY2_MULTITASK.ipynb": ("Study 2 - Multi-task and Multi-model", "study2", None),
    "03_STUDY3_KOREAN.ipynb": ("Study 3 - Korean Generalization", "study3", None),
    "04_AGGREGATE.ipynb": ("Aggregate Paper Results", None, "aggregate"),
}

out = ROOT / "notebooks"; out.mkdir(parents=True, exist_ok=True)
for name, (title, study, action) in specs.items():
    (out / name).write_text(json.dumps(notebook(title, study, action), ensure_ascii=False, indent=1), encoding="utf-8")
(out / "05_PROGRESS_MONITOR.ipynb").write_text(json.dumps(monitor_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
print(f"created {len(specs) + 1} notebooks in {out}")
