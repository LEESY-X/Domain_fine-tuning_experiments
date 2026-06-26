# 실험 개요

본 문서는 논문 부록(supplementary material) 관점에서 fine-tuning strategy 실험의 목적, 방법, 데이터셋, 평가 방식, 결과 해석을 정리한다. 문서의 모든 수치는 기존 실험 산출물에서 확인 가능한 값만 사용하였다.

## 1. 연구 목적

본 프로젝트의 목적은 사전학습 언어모델을 텍스트 분류 문제에 적용할 때, Full Fine-tuning과 parameter-efficient fine-tuning(PEFT) 계열 방법이 성능, 학습 시간, 학습 파라미터 수, seed 안정성 측면에서 어떻게 달라지는지 비교하는 것이다.

비교 대상 방법은 다음 5개이다.

| 방법 | 설명 | 출처 파일 |
|---|---|---|
| Full Fine-tuning | 전체 모델 파라미터를 업데이트하는 기준선 | `config/experiment_config.json`, `src/suite.py` |
| LoRA | `query`, `value` target module에 low-rank adapter를 적용 | `config/experiment_config.json`, `src/suite.py` |
| Adapter | Transformer layer output에 bottleneck adapter를 삽입 | `config/experiment_config.json`, `src/suite.py` |
| IA3 | `key`, `value`, `intermediate.dense` 계열 module을 대상으로 IA3 설정 적용 | `src/suite.py` |
| BitFit | bias parameter와 classification head 중심으로 학습 | `src/suite.py` |

## 2. 실험 구성

최종 요약 테이블 기준으로 전체 실험은 3개 Study, 22개 task/model 조합, 5개 method, 5개 seed로 구성된다.

| Study | 모델 | 데이터셋/task | Epoch | Run 수 | 출처 파일 |
|---|---|---|---:|---:|---|
| Study 1 | `vinai/bertweet-base` | `measuring_hate_speech` | 3 | 25 | `config/experiment_config.json`, `final_tables/summary_by_task_model_method.csv` |
| Study 2 | `vinai/bertweet-base`, `FacebookAI/roberta-base` | 9개 영어 task | 2 | 450 | `config/experiment_config.json`, `final_tables/summary_by_task_model_method.csv` |
| Study 3 | `klue/roberta-base` | `news_ynat`, `movie_nsmc`, `comment_kmhas_binary` | 5 | 75 | `config/experiment_config.json`, `final_tables/summary_by_task_model_method.csv` |
| 전체 | 3개 base model | 22개 task/model 조합 | Study별 상이 | 550 | 원본 workspace의 `final_tables/all_runs_550.csv`; 업로드 폴더에서는 run-level table 제외 |

## 3. 모델 및 tokenizer

사용 모델은 `config/experiment_config.json`과 최종 집계 테이블에서 확인된 다음 3개이다.

| 모델 | 사용 Study | tokenizer 확인 내용 | 출처 파일 |
|---|---|---|---|
| `vinai/bertweet-base` | Study 1, Study 2 | `AutoTokenizer.from_pretrained(..., use_fast=False if "bertweet" in model_name.lower() else True)` | `src/suite.py` |
| `FacebookAI/roberta-base` | Study 2 | `AutoTokenizer.from_pretrained` 사용 | `src/suite.py` |
| `klue/roberta-base` | Study 3 | `AutoTokenizer.from_pretrained` 사용 | `src/suite.py` |

checkpoint 및 output model은 `results/**/checkpoints/` 하위에 저장되어 있다. 공개 저장소 업로드 전에는 대용량 checkpoint를 Git LFS 또는 외부 저장소로 분리해야 한다.

## 4. 데이터셋

데이터셋 출처와 컬럼 정의는 `src/suite.py`의 `TASKS`에서 확인된다.

| Task | Source | Subset/direct | Text column | Label column | Labels | 출처 파일 |
|---|---|---|---|---|---:|---|
| `measuring_hate_speech` | `ucberkeley-dlab/measuring-hate-speech` | `train` split 기반 | `comment` | `hatespeech` | 2 | `src/suite.py` |
| `tweet_sentiment` | `cardiffnlp/tweet_eval` | `sentiment` | `text` | `label` | 3 | `src/suite.py` |
| `finance_sentiment` | `lmassaron/FinancialPhraseBank` | 확인 필요 | `sentence` | `label` | 3 | `src/suite.py` |
| `movie_reviews` | `stanfordnlp/imdb` | 확인 필요 | `text` | `label` | 2 | `src/suite.py` |
| `product_reviews` | `SetFit/amazon_reviews_multi_en` | 확인 필요 | `text` | `label` | 5 | `src/suite.py` |
| `tweet_emotion` | `cardiffnlp/tweet_eval` | `emotion` | `text` | `label` | 4 | `src/suite.py` |
| `tweet_hate` | `cardiffnlp/tweet_eval` | `hate` | `text` | `label` | 2 | `src/suite.py` |
| `tweet_offensive` | `cardiffnlp/tweet_eval` | `offensive` | `text` | `label` | 2 | `src/suite.py` |
| `tweet_irony` | `cardiffnlp/tweet_eval` | `irony` | `text` | `label` | 2 | `src/suite.py` |
| `news_topic` | `fancyzhx/ag_news` | 확인 필요 | `text` | `label` | 4 | `src/suite.py` |
| `news_ynat` | `klue` | `ynat` | `title` | `label` | 7 | `src/suite.py` |
| `movie_nsmc` | raw GitHub TSV via `e9t/nsmc` | `direct="nsmc"` | `document` | `label` | 2 | `src/suite.py` |
| `comment_kmhas_binary` | raw GitHub TSV via `adlnlp/K-MHaS` | `direct="kmhas"` | `text` | `label` | 2 | `src/suite.py` |

데이터 공개 가능 여부는 각 원천 데이터셋의 라이선스와 Hugging Face/GitHub 원천 약관을 별도로 확인해야 한다. 본 저장소에는 cache와 checkpoint가 포함되어 있어 공개 전 데이터 재배포 가능성 검토가 필요하다.

## 5. 평가 방식

평가는 test split에 대한 `Trainer.predict` 결과를 사용하며, `src/suite.py`의 `compute_metrics`에서 다음 지표가 계산된다.

| Metric | 의미 | 출처 파일 |
|---|---|---|
| `accuracy` | 전체 sample 중 정답 비율 | `src/suite.py` |
| `macro_f1` | class별 F1의 macro average | `src/suite.py` |
| `macro_precision` | class별 precision의 macro average | `src/suite.py` |
| `macro_recall` | class별 recall의 macro average | `src/suite.py` |

최종 요약 문서는 `macro_f1`, 학습 시간(`train_seconds_mean`), 학습 파라미터 비율(`trainable_ratio_mean`), seed별 표준편차(`f1_sd`)를 중심으로 결과를 해석한다.

## 6. 주요 결과

최종 결과 파일 기준 주요 집계는 다음과 같다.

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| 완료 run 수 | 550 | 원본 workspace의 `final_tables/all_runs_550.csv`; 업로드 폴더에서는 run-level table 제외 |
| method별 summary row 수 | 110 | `final_tables/summary_by_task_model_method.csv` |
| task/model 조합 수 | 22 | `final_tables/summary_by_task_model_method.csv` |
| 성능 1위: Full Fine-tuning | 19 / 22 | `final_tables/winners_by_metric.csv` |
| 성능 1위: IA3 | 2 / 22 | `final_tables/winners_by_metric.csv` |
| 성능 1위: Adapter | 1 / 22 | `final_tables/winners_by_metric.csv` |
| 평균 Macro-F1: Full Fine-tuning | 0.7588 | `final_tables/summary_by_task_model_method.csv` |
| 평균 Macro-F1: IA3 | 0.6702 | `final_tables/summary_by_task_model_method.csv` |
| 평균 Macro-F1: LoRA | 0.6537 | `final_tables/summary_by_task_model_method.csv` |
| 평균 Macro-F1: Adapter | 0.6506 | `final_tables/summary_by_task_model_method.csv` |
| 평균 Macro-F1: BitFit | 0.6443 | `final_tables/summary_by_task_model_method.csv` |

해석상 중요한 점은 Full Fine-tuning이 전반적으로 강한 기준선이지만, 일부 hate-speech task에서는 IA3가 Full FT보다 높은 `macro_f1`을 보였고, Study 3의 `news_ynat`에서는 Adapter가 성능 1위로 집계되었다는 점이다.

## 7. 재현성 및 공개 주의사항

이 저장소는 원본 노트북과 결과 파일을 수정하지 않는 방식으로 결과 확인을 지원한다. 논문 첨부용 GitHub 저장소로 공개할 때는 다음 항목을 사람이 직접 확인해야 한다.

- `results/**/checkpoints/` 하위 대용량 checkpoint 파일 포함 여부
- `cache/` 하위 데이터셋 cache 재배포 가능 여부
- 로컬 절대경로 및 개인 계정명 노출 여부
- 원천 데이터셋 라이선스
- GitHub 단일 파일 100MB 제한 및 Git LFS 적용 여부
