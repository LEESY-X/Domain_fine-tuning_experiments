# Results Table

본 문서는 최종 결과 파일에서 확인 가능한 값만 표로 정리한다. 결과값은 새로 계산하거나 재학습하지 않고 기존 CSV를 읽어 확인하였다.

## 1. 전체 실험 규모

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| 전체 run 수 | 550 | 원본 workspace의 `final_tables/all_runs_550.csv`; 업로드 폴더에서는 run-level table 제외 |
| summary row 수 | 110 | `final_tables/summary_by_task_model_method.csv` |
| task/model 조합 수 | 22 | `final_tables/summary_by_task_model_method.csv` |
| 비교 method 수 | 5 | `config/experiment_config.json` |
| seed 수 | 5 | `config/experiment_config.json` |

## 2. Method별 평균 Macro-F1

| Method | 평균 Macro-F1 | 출처 파일 |
|---|---:|---|
| Full Fine-tuning | 0.7588 | `final_tables/summary_by_task_model_method.csv` |
| IA3 | 0.6702 | `final_tables/summary_by_task_model_method.csv` |
| LoRA | 0.6537 | `final_tables/summary_by_task_model_method.csv` |
| Adapter | 0.6506 | `final_tables/summary_by_task_model_method.csv` |
| BitFit | 0.6443 | `final_tables/summary_by_task_model_method.csv` |

## 3. 성능 1위 분포

| 성능 1위 method | 횟수 | 출처 파일 |
|---|---:|---|
| Full Fine-tuning | 19 | `final_tables/winners_by_metric.csv` |
| IA3 | 2 | `final_tables/winners_by_metric.csv` |
| Adapter | 1 | `final_tables/winners_by_metric.csv` |
| LoRA | 0 | `final_tables/winners_by_metric.csv` |
| BitFit | 0 | `final_tables/winners_by_metric.csv` |

## 4. 파라미터 효율 1위 분포

`winners_by_metric.csv`의 `most_efficient_method`는 `efficient_ratio` 기준 최저 trainable parameter ratio를 의미한다.

| 효율 1위 method | 횟수 | 출처 파일 |
|---|---:|---|
| IA3 | 22 | `final_tables/winners_by_metric.csv` |

## 5. 안정성 1위 분포

`most_stable_method`는 seed별 `f1_sd`가 가장 낮은 method를 의미한다.

| 안정성 1위 method | 횟수 | 출처 파일 |
|---|---:|---|
| BitFit | 10 | `final_tables/winners_by_metric.csv` |
| LoRA | 4 | `final_tables/winners_by_metric.csv` |
| IA3 | 4 | `final_tables/winners_by_metric.csv` |
| Full Fine-tuning | 2 | `final_tables/winners_by_metric.csv` |
| Adapter | 2 | `final_tables/winners_by_metric.csv` |

## 6. Task/model별 성능 1위

| Study | Task | Model | Best method | Macro-F1 | 출처 파일 |
|---|---|---|---|---:|---|
| study1 | measuring_hate_speech | vinai/bertweet-base | full_ft | 0.8148 | `final_tables/winners_by_metric.csv` |
| study2 | finance_sentiment | FacebookAI/roberta-base | full_ft | 0.8364 | `final_tables/winners_by_metric.csv` |
| study2 | finance_sentiment | vinai/bertweet-base | full_ft | 0.7616 | `final_tables/winners_by_metric.csv` |
| study2 | movie_reviews | FacebookAI/roberta-base | full_ft | 0.9108 | `final_tables/winners_by_metric.csv` |
| study2 | movie_reviews | vinai/bertweet-base | full_ft | 0.8954 | `final_tables/winners_by_metric.csv` |
| study2 | news_topic | FacebookAI/roberta-base | full_ft | 0.9502 | `final_tables/winners_by_metric.csv` |
| study2 | news_topic | vinai/bertweet-base | full_ft | 0.9444 | `final_tables/winners_by_metric.csv` |
| study2 | product_reviews | FacebookAI/roberta-base | full_ft | 0.6081 | `final_tables/winners_by_metric.csv` |
| study2 | product_reviews | vinai/bertweet-base | full_ft | 0.6088 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_emotion | FacebookAI/roberta-base | full_ft | 0.7081 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_emotion | vinai/bertweet-base | full_ft | 0.6643 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_hate | FacebookAI/roberta-base | ia3 | 0.5727 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_hate | vinai/bertweet-base | ia3 | 0.5349 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_irony | FacebookAI/roberta-base | full_ft | 0.6005 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_irony | vinai/bertweet-base | full_ft | 0.7366 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_offensive | FacebookAI/roberta-base | full_ft | 0.8003 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_offensive | vinai/bertweet-base | full_ft | 0.8054 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_sentiment | FacebookAI/roberta-base | full_ft | 0.7106 | `final_tables/winners_by_metric.csv` |
| study2 | tweet_sentiment | vinai/bertweet-base | full_ft | 0.7247 | `final_tables/winners_by_metric.csv` |
| study3 | comment_kmhas_binary | klue/roberta-base | full_ft | 0.8874 | `final_tables/winners_by_metric.csv` |
| study3 | movie_nsmc | klue/roberta-base | full_ft | 0.9096 | `final_tables/winners_by_metric.csv` |
| study3 | news_ynat | klue/roberta-base | adapter | 0.8702 | `final_tables/winners_by_metric.csv` |
