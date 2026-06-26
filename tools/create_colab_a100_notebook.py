import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "COLAB_A100_FULL_550_RUNS.ipynb"
suite_source = (ROOT / "src" / "suite.py").read_text(encoding="utf-8")
config_source = (ROOT / "config" / "experiment_config.json").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in text.strip().splitlines()]}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [x + "\n" for x in text.strip().splitlines()]}


cells = [
    md("""
# Colab A100 — 5-Method Full Paper Experiment (550 Runs)

이 노트북 하나로 로컬 실험과 동일한 **데이터셋·모델·5개 방법·5개 seed·epoch·FP16·effective batch 64** 조건을 Google Colab A100에서 실행합니다.

## 중요

- Colab 런타임을 반드시 **A100 GPU**로 선택합니다.
- 결과와 체크포인트는 Google Drive의 `MyDrive/paper_finetuning_5method_A100`에 저장됩니다.
- 연결이 끊기면 같은 Study 셀을 다시 실행합니다. 완료 run은 건너뛰고 중단 run은 마지막 정상 epoch checkpoint부터 재개합니다.
- 동일 Drive 폴더를 두 Colab 세션에서 동시에 실행하지 마세요.
- GPU가 다르므로 학습시간과 부동소수점 결과가 로컬과 비트 단위로 같지는 않지만, 논문 실험 프로토콜은 동일합니다.
"""),
    md("""
## 1. 패키지 설치

로컬에서 검증한 Transformers·Datasets·PEFT 버전을 설치합니다. Colab의 CUDA 호환 PyTorch는 그대로 사용합니다.
"""),
    code("""
%pip install -q "transformers==5.9.0" "datasets==4.8.5" "peft==0.19.1" accelerate scikit-learn pandas numpy sentencepiece
"""),
    md("""
## 2. Google Drive 연결 및 단일 노트북 코드 배치

다음 셀은 이 노트북에 내장된 실행 엔진과 설정을 Drive에 기록합니다. `results` 폴더는 삭제하거나 덮어쓰지 않습니다.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, sys

DRIVE_ROOT = Path('/content/drive/MyDrive/paper_finetuning_5method_A100')
(DRIVE_ROOT / 'src').mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / 'config').mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / 'src' / '__init__.py').write_text('', encoding='utf-8')
print('DRIVE_ROOT =', DRIVE_ROOT)
"""),
    code("SUITE_SOURCE = " + repr(suite_source) + "\n(DRIVE_ROOT / 'src' / 'suite.py').write_text(SUITE_SOURCE, encoding='utf-8')\nprint('suite.py written:', len(SUITE_SOURCE), 'chars')"),
    code("CONFIG_SOURCE = " + repr(config_source) + "\n(DRIVE_ROOT / 'config' / 'experiment_config.json').write_text(CONFIG_SOURCE, encoding='utf-8')\nprint(CONFIG_SOURCE)"),
    md("""
## 3. A100 및 프로토콜 사전점검

GPU 이름이 A100이 아니면 실행을 중단합니다. 로컬과 동일하게 FP16을 사용하며 A100이라고 BF16이나 batch size를 변경하지 않습니다.
"""),
    code("""
sys.path.insert(0, str(DRIVE_ROOT))
from src.suite import precheck, load_config, build_jobs, run_study, aggregate

info = precheck(require_cuda=True)
display(info)
if 'A100' not in info['gpu'].upper():
    raise RuntimeError(f"A100 런타임이 아닙니다: {info['gpu']}")

cfg = load_config()
assert cfg['methods'] == ['full_ft', 'lora', 'adapter', 'ia3', 'bitfit']
assert cfg['seeds'] == [42, 52, 62, 72, 82]
assert cfg['precision'] == 'fp16'
assert cfg['batch_size'] == 16
assert cfg['gradient_accumulation_steps'] == 4
assert cfg['study2']['limits'] is None
print('PROTOCOL CHECK PASS')
print('Study 1 jobs:', len(build_jobs('study1')))
print('Study 2 jobs:', len(build_jobs('study2')))
print('Study 3 jobs:', len(build_jobs('study3')))
"""),
    md("""
## 4. Study 1 실행 — 25 Runs

BERTweet + Measuring Hate Speech, 5 methods × 5 seeds. 예상 A100 시간은 로컬보다 짧지만 Drive 저장시간에 따라 달라질 수 있습니다.
"""),
    code("""
STUDY = 'study1'
RUN_MODE = 'PAPER'
study1_result = run_study(STUDY, run_mode=RUN_MODE, max_jobs=None, continue_on_error=False)
display(study1_result.tail())
"""),
    md("""
## 5. Study 3 실행 — 75 Runs

KLUE-RoBERTa + YNAT/NSMC/K-MHaS, 5 methods × 5 seeds.
"""),
    code("""
STUDY = 'study3'
RUN_MODE = 'PAPER'
study3_result = run_study(STUDY, run_mode=RUN_MODE, max_jobs=None, continue_on_error=False)
display(study3_result.tail())
"""),
    md("""
## 6. Study 2 실행 — 450 Runs

9 English tasks × 2 models × 5 methods × 5 seeds. 원본 split 전체를 사용합니다.
"""),
    code("""
STUDY = 'study2'
RUN_MODE = 'PAPER'
study2_result = run_study(STUDY, run_mode=RUN_MODE, max_jobs=None, continue_on_error=False)
display(study2_result.tail())
"""),
    md("""
## 7. 전체 결과 집계

세 Study가 완료된 뒤 실행합니다.
"""),
    code("""
all_runs = aggregate('PAPER')
display(all_runs.head())
print('completed result rows:', len(all_runs), '/ 550')
assert len(all_runs) == 550, '아직 완료되지 않은 run이 있습니다.'
"""),
    md("""
## 8. 진행상태 확인

학습 셀이 중단된 뒤 또는 Study 사이에 실행할 수 있습니다.
"""),
    code("""
from collections import Counter

for study in ('study1', 'study3', 'study2'):
    root = DRIVE_ROOT / 'results' / study / 'PAPER'
    statuses = []
    for path in root.glob('**/status.json') if root.exists() else []:
        try:
            statuses.append(json.loads(path.read_text(encoding='utf-8')).get('status', 'UNKNOWN'))
        except Exception:
            statuses.append('UNREADABLE')
    print(study, dict(Counter(statuses)))
    progress = root / 'progress.json'
    if progress.exists():
        print(progress.read_text(encoding='utf-8'))
"""),
    md("""
## 재개 방법

1. Colab 연결이 끊기면 1~3번 셀을 다시 실행합니다.
2. 중단된 Study의 실행 셀을 다시 실행합니다.
3. `COMPLETE` run은 자동으로 건너뜁니다.
4. 중단 run은 Google Drive에 저장된 마지막 정상 epoch checkpoint에서 재개합니다.
5. 오류 발생 시 해당 run 폴더의 `error.txt`와 `status.json`을 확인합니다.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
