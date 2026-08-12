# Experiment Summary

본 문서는 논문 독자가 실험 구조와 결과를 빠르게 이해할 수 있도록 작성한 요약이다. 원본 노트북, 결과 CSV, JSON, checkpoint는 수정하지 않았다.

## 연구 질문

본 실험은 텍스트 분류 문제에서 Full Fine-tuning과 PEFT 방법들이 다음 기준에서 어떻게 다른지 비교한다.

- Macro-F1 기준 성능
- 학습 시간
- 학습 가능한 parameter 비율
- seed별 안정성

## 실험 설계

최종 실험은 3개 Study로 구성된다.

| Study | 목적 | 구성 | 출처 파일 |
|---|---|---|---|
| Study 1 | BERTweet 기반 단일 hate-speech task에서 method 간 차이 확인 | `vinai/bertweet-base` + `measuring_hate_speech` | `config/experiment_config.json` |
| Study 2 | 영어 task와 base model을 확장하여 일반화 양상 확인 | 9개 task x 2개 model | `config/experiment_config.json` |
| Study 3 | 한국어 task에서 동일한 비교 구조 확인 | `klue/roberta-base` + 3개 한국어 task | `config/experiment_config.json` |

전체 run 수는 원본 workspace의 `final_tables/all_runs_550.csv` 기준 550개로 확인하였다. 다만 GitHub 업로드 폴더에는 매 run 단위 table을 포함하지 않고, 최종 요약표만 포함한다.

## 주요 관찰

1. Full Fine-tuning은 평균 Macro-F1과 성능 1위 빈도에서 가장 강한 기준선이다.
2. IA3는 전체 평균 성능에서는 Full Fine-tuning보다 낮지만, `tweet_hate` task의 두 model 조건에서 성능 1위로 집계되었다.
3. Adapter는 Study 3의 `news_ynat`에서 성능 1위로 집계되었고, 일부 task에서 Full Fine-tuning에 근접한 성능을 보였다.
4. LoRA는 trainable parameter ratio를 크게 줄이지만, wall-clock time까지 항상 크게 줄이는 것은 아니다.
5. 표준편차만으로 안정성을 평가하면 단일 클래스 예측 붕괴를 안정성으로 오인할 수 있다. 후속 분석은 예측 클래스 수와 최빈 예측 비율을 함께 보고한다.

## 해석상 주의사항

- Study별 데이터셋, 모델, epoch 수가 다르므로 학습시간은 Study 간 절대 비교보다 동일 task/model 내부 비교로 해석해야 한다.
- `SMOKE` 결과는 최종 논문 결론에 포함하지 않는다.
- 공개용 경량 repository에는 `cache/`와 `results/**/checkpoints/`를 포함하지 않는다. 데이터셋은 코드에서 원천 source를 다시 불러오는 구조로 정리하였다.
- 현재 재현 코드는 optimizer=`adamw_torch`, scheduler=`linear`를 설정 파일에 명시한다. 원본 550회 run의 run-level 설정 파일은 공개 패키지에 없으므로, 원본 실행이 동일 기본값을 사용했는지는 별도 원시 증거 없이 단정하지 않는다.
- 원본의 exact-zero SD 네 행은 `results_summary/variance_diagnostics.md`에서 constant-class metric fingerprint로 진단했으며, 후속 재학습은 `results_summary/paper_followup/`에 정리한다.

## 재현성과 논문 작성 자료

- `REPRODUCIBILITY_RECORD.md`: 원본 550회와 후속 100회의 설정, 환경, 데이터 fingerprint, 실행 명령, 통계 정의, checksum, 공개 범위와 남은 재현성 격차를 한 파일에 정리한다.
- `PAPER_CONTENT_CANDIDATES.md`: Abstract부터 Appendix까지 삽입 가능한 결과, 표·그림 후보, 추가실험, limitations, ethics, 주장 금지 항목과 제출 전 체크리스트를 정리한다.
