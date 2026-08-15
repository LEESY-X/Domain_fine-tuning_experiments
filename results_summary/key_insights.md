# Key Insights

본 문서는 최종 요약표에서 확인 가능한 결과를 바탕으로 논문 discussion에 사용할 수 있는 핵심 관찰을 정리한다. 해석은 `final_tables/summary_by_task_model_method.csv`와 `final_tables/winners_by_metric.csv`에서 확인 가능한 범위로 제한하였다.

## 1. Full Fine-tuning remains the strongest general baseline

22개 task/model 조합 중 19개에서 Full Fine-tuning이 Macro-F1 기준 성능 1위로 집계되었다. method별 평균 Macro-F1도 Full Fine-tuning이 0.7588로 가장 높았다.

| Method | 평균 Macro-F1 | 성능 1위 횟수 | 출처 파일 |
|---|---:|---:|---|
| Full Fine-tuning | 0.7588 | 19 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |
| IA3 | 0.6702 | 2 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |
| LoRA | 0.6537 | 0 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |
| Adapter | 0.6506 | 1 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |
| BitFit | 0.6443 | 0 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |

## 2. PEFT can outperform Full FT in specific task conditions

Full Fine-tuning이 전반적으로 강하지만, 일부 task에서는 PEFT 방법이 성능 1위로 집계되었다. 특히 `tweet_hate`에서는 IA3가 두 base model 조건 모두에서 성능 1위였고, `news_ynat`에서는 Adapter가 성능 1위였다.

| Study | Task | Model | Best method | Macro-F1 | 출처 파일 |
|---|---|---|---|---:|---|
| study2 | tweet_hate | FacebookAI/roberta-base | IA3 | 0.5727 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_hate | vinai/bertweet-base | IA3 | 0.5349 | `final_tables/winners_by_metric.csv` |
| study3 | news_ynat | klue/roberta-base | Adapter | 0.8702 | `final_tables/winners_by_metric.csv` |

## 3. Parameter efficiency and wall-clock efficiency are different questions

`winners_by_metric.csv` 기준으로 trainable parameter ratio가 가장 낮은 방법은 22개 task/model 조합 모두에서 IA3였다. 그러나 이 결과가 곧 모든 조건에서 가장 빠른 wall-clock training을 의미하지는 않는다. 본 연구에서는 parameter efficiency, performance, stability를 분리해서 해석해야 한다.

## 4. Lowest SD alone is not a valid stability criterion

기존 `winners_by_metric.csv`에서 BitFit은 10/22 조건의 최저 Macro-F1 SD 방법으로 표시되지만, 그중 일부는 모든 seed가 같은 단일 클래스를 예측하여 SD가 0이 된 경우다. 따라서 “최저 SD”를 곧바로 바람직한 seed 안정성으로 해석해서는 안 된다.

constant-class fingerprint가 확인된 4개 방법 행을 안정성 후보에서 제외하면 최저 SD 방법의 빈도는 아래와 같이 바뀐다. 이 값도 예측 분포를 직접 확인하지 못한 나머지 원본 run에는 잠재적 붕괴가 없다는 보장은 아니므로 sensitivity analysis로만 사용한다.

| Stable method after excluding diagnosed collapse | 조건 수 | 출처 파일 |
|---|---:|---|
| BitFit | 8 | `final_tables/summary_by_task_model_method.csv` |
| IA3 | 5 | `final_tables/summary_by_task_model_method.csv` |
| Full Fine-tuning | 3 | `final_tables/summary_by_task_model_method.csv` |
| LoRA | 3 | `final_tables/summary_by_task_model_method.csv` |
| Adapter | 3 | `final_tables/summary_by_task_model_method.csv` |

## 5. Practical recommendation

결과는 하나의 universal best method보다 목적별 선택 기준이 더 적절하다는 점을 시사한다.

| 목적 | 우선 검토할 방법 | 근거 |
|---|---|---|
| 최고 성능 기준선 | Full Fine-tuning | 성능 1위 19/22, 평균 Macro-F1 0.7588 |
| 특정 hate-speech task의 PEFT 대안 | IA3 | `tweet_hate` 2개 model 조건에서 성능 1위 |
| 한국어 news topic task 대안 | Adapter | `news_ynat`에서 성능 1위 |
| trainable parameter 최소화 | IA3 | 22/22 task/model 조합에서 최저 trainable ratio |
| seed 안정성 우선 | 예측 분포 진단 후 선택 | SD만으로 순위를 정하면 constant-class collapse를 안정성으로 오인할 수 있음 |

후속 진단과 재학습 결과의 논문용 표·주장 범위는 `results_summary/paper_followup/`에 별도로 정리한다.
