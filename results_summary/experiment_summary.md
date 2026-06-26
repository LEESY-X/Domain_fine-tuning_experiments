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
5. BitFit은 안정성 1위 빈도가 가장 높지만, 복잡한 task에서는 성능 손실이 발생할 수 있다.

## 해석상 주의사항

- Study별 데이터셋, 모델, epoch 수가 다르므로 학습시간은 Study 간 절대 비교보다 동일 task/model 내부 비교로 해석해야 한다.
- `SMOKE` 결과는 최종 논문 결론에 포함하지 않는다.
- 공개 저장소에는 `cache/`와 `results/**/checkpoints/`가 포함될 수 있으므로, 데이터 재배포 가능성과 대용량 파일 정책을 반드시 별도로 확인해야 한다.
- optimizer와 scheduler의 명시적 설정은 현재 확인 필요로 남아 있다. `TrainingArguments` 기본값이 적용되었을 수 있으나, 본 문서에서는 추정하지 않는다.
