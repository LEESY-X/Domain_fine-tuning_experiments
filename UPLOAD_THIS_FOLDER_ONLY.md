# Upload This Folder Only

이 폴더는 논문 첨부용 GitHub repository에 올릴 파일만 따로 복사한 경량 업로드 폴더이다. 목적은 매 run마다 저장된 데이터 전체를 공개하는 것이 아니라, 실험 재현에 필요한 코드, 설정, 노트북, 최종 요약표를 제공하는 것이다.

원본 실험 workspace 전체는 약 269 GB이며, `cache/`, `results/**/checkpoints/`, 전체 run-level checkpoint와 dataset cache를 포함하므로 GitHub에 직접 업로드하면 안 된다.

## 포함한 항목

- 논문 supplementary repository용 `README.md`
- package 목록: `requirements.txt`
- 공개용 ignore 규칙: `.gitignore`
- 실험 설정: `config/experiment_config.json`
- 최종 요약 결과: `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv`
- 실행/분석 notebook: `notebooks/*.ipynb`, `COLAB_A100_*.ipynb`
- 핵심 실행 코드: `src/`, `tools/`
- 환경 기록: `results/environment.json`
- 논문 보조 문서: `docs/`, `results_summary/`

## 제외한 항목

- `cache/`
- `results/study*/.../checkpoints/`
- `results/study*/.../predictions.csv`
- `results/study*/.../trainer_history.csv`
- `results/study*/.../events.jsonl`
- `results/study*/.../final_metrics.json`
- `results/study*/.../epoch_metrics.csv`
- `final_tables/all_runs_550.csv`
- 별도 보고서 산출물
- `__pycache__/`
- `.ipynb_checkpoints/`
- 로컬 작업 이어받기 문서
- 로컬 절대경로가 포함된 Notion 작업 가이드

## 업로드 전 확인

1. GitHub repository를 새로 만든다.
2. 이 폴더 안의 파일만 업로드한다.
3. `README.md`의 Citation placeholder를 논문 정보와 GitHub URL로 수정한다.
4. 원천 데이터셋 라이선스와 checkpoint 공개 여부를 별도로 확인한다.
