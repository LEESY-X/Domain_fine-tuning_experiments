# Fine-Tuning Strategy Reproducibility Repository

본 저장소는 텍스트 분류 과제에서 Full Fine-tuning과 parameter-efficient fine-tuning(PEFT) 전략을 비교한 논문 보조 자료(supplementary repository)이다. 원본 Jupyter Notebook, 실험 결과, metric, log, checkpoint, config 파일은 재현성 확인을 위한 읽기 전용 근거로 유지한다.

## Abstract

본 레포지토리는 사전학습 언어모델을 다양한 텍스트 분류 데이터셋에 fine-tuning할 때, Full Fine-tuning, LoRA, Adapter, IA3, BitFit이 성능, 학습 시간, 학습 파라미터 수, seed 안정성 측면에서 어떻게 다른지 비교하기 위한 실험 자료를 제공한다.

실험은 `vinai/bertweet-base`, `FacebookAI/roberta-base`, `klue/roberta-base`를 대상으로 수행되었으며, 영어 social media, sentiment, hate speech, review, news topic task와 한국어 news, movie review, hate/comment classification task를 포함한다. 최종 결과 파일 기준 전체 실험은 3개 Study, 22개 task/model 조합, 5개 method, 5개 seed, 총 550개 run으로 구성된다.

주요 결과는 Full Fine-tuning이 평균 Macro-F1과 성능 1위 빈도에서 가장 강한 기준선이라는 점이다. 그러나 IA3는 일부 hate-speech task에서 Full FT보다 높은 성능을 보였고, Adapter는 `news_ynat`에서 성능 1위로 집계되었다. 따라서 본 실험은 단일한 최적 fine-tuning 방식보다, 목표 지표와 task 특성에 따른 method selection의 필요성을 보여준다.

## 1. Introduction

사전학습 언어모델을 실제 분류 문제에 적용할 때 가장 직접적인 접근은 Full Fine-tuning이다. Full Fine-tuning은 모든 파라미터를 업데이트하므로 일반적으로 강한 성능 기준선이 되지만, 학습 비용과 저장 비용이 크다. 반면 LoRA, Adapter, IA3, BitFit과 같은 PEFT 방법은 일부 파라미터만 학습하여 비용을 줄일 수 있으나, task와 model 조건에 따라 성능 손실 또는 시간 효율 저하가 발생할 수 있다.

본 프로젝트는 이러한 trade-off를 동일한 실험 구조 안에서 비교하기 위해 구성되었다. 이 저장소의 역할은 논문 독자가 실험 구성, hyperparameter, 평가 지표, 결과 파일, 공개 전 주의사항을 확인할 수 있도록 원본 실험 산출물과 요약 문서를 제공하는 것이다.

## 2. Repository Structure

현재 구조를 기준으로 주요 파일과 폴더는 다음과 같다. 기존 실험 파일은 이동하지 않았다.

```text
.
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.json
├── notebooks/
│   ├── 00_PRECHECK.ipynb
│   ├── 01_STUDY1_BERTWEET_HATE.ipynb
│   ├── 02_STUDY2_MULTITASK.ipynb
│   ├── 03_STUDY3_KOREAN.ipynb
│   ├── 04_AGGREGATE.ipynb
│   └── 05_PROGRESS_MONITOR.ipynb
├── src/
│   └── suite.py
├── scripts/
│   └── build_*.py
├── tools/
│   └── *.py
├── final_tables/
│   ├── summary_by_task_model_method.csv
│   └── winners_by_metric.csv
├── results/
│   ├── environment.json
│   ├── aggregate/
│   ├── study1/
│   ├── study2/
│   └── study3/
├── docs/
│   └── experiment_overview.md
├── results_summary/
│   ├── experiment_summary.md
│   ├── hyperparameters.md
│   ├── key_insights.md
│   └── results_table.md
├── github_assets/
│   └── README.md
└── results/
    └── environment.json
```

## 3. Model

| Base model | 사용 범위 | Tokenizer 정보 | Fine-tuning 방식 | Checkpoint/output 위치 | 출처 파일 |
|---|---|---|---|---|---|
| `vinai/bertweet-base` | Study 1, Study 2 | `AutoTokenizer.from_pretrained`, BERTweet일 때 `use_fast=False` | Full FT, LoRA, Adapter, IA3, BitFit | `results/study1/`, `results/study2/` | `config/experiment_config.json`, `src/suite.py` |
| `FacebookAI/roberta-base` | Study 2 | `AutoTokenizer.from_pretrained` | Full FT, LoRA, Adapter, IA3, BitFit | `results/study2/` | `config/experiment_config.json`, `src/suite.py` |
| `klue/roberta-base` | Study 3 | `AutoTokenizer.from_pretrained` | Full FT, LoRA, Adapter, IA3, BitFit | `results/study3/` | `config/experiment_config.json`, `src/suite.py` |

학습 과정에서 생성된 checkpoint는 공개용 경량 repository에 포함하지 않았다. 본 repository는 code, configuration, notebook, environment record, final summary tables 중심으로 구성하였다.

## 4. Dataset

데이터셋 정의는 `src/suite.py`의 `TASKS`에서 확인된다.

| Task | Source | Subset/direct | Train | Validation | Test | 공개 가능 여부 | 출처 파일 |
|---|---|---|---:|---:|---:|---|---|
| `measuring_hate_speech` | `ucberkeley-dlab/measuring-hate-speech` | `train` 기반 split | 108,444 | 13,556 | 13,556 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `finance_sentiment` | `lmassaron/FinancialPhraseBank` | 확인 필요 | 3,872 | 484 | 484 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `movie_reviews` | `stanfordnlp/imdb` | 확인 필요 | 22,500 | 2,500 | 25,000 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `news_topic` | `fancyzhx/ag_news` | 확인 필요 | 108,000 | 12,000 | 7,600 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `product_reviews` | `SetFit/amazon_reviews_multi_en` | 확인 필요 | 200,000 | 5,000 | 5,000 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `tweet_emotion` | `cardiffnlp/tweet_eval` | `emotion` | 3,257 | 374 | 1,421 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `tweet_hate` | `cardiffnlp/tweet_eval` | `hate` | 9,000 | 1,000 | 2,970 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `tweet_irony` | `cardiffnlp/tweet_eval` | `irony` | 2,862 | 955 | 784 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `tweet_offensive` | `cardiffnlp/tweet_eval` | `offensive` | 11,916 | 1,324 | 860 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `tweet_sentiment` | `cardiffnlp/tweet_eval` | `sentiment` | 45,615 | 2,000 | 12,284 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `comment_kmhas_binary` | raw GitHub TSV via `adlnlp/K-MHaS` | `direct="kmhas"` | 78,977 | 8,776 | 21,939 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `movie_nsmc` | raw GitHub TSV via `e9t/nsmc` | `direct="nsmc"` | 134,995 | 15,000 | 49,997 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |
| `news_ynat` | `klue` | `ynat` | 41,110 | 4,568 | 9,107 | 공개 가능 여부 확인 필요 | `src/suite.py`, `final_tables/summary_by_task_model_method.csv` |

전처리는 `src/suite.py`의 `_standardize`, `_ensure_three_splits`, tokenizer mapping 단계에서 수행된다. `SMOKE` mode는 `config/experiment_config.json` 기준 train/validation/test를 128/64/64로 제한하므로 논문 결과 해석에는 포함하지 않는다.

## 5. Experimental Setup

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| Python | `3.10.19` | `results/environment.json` |
| PyTorch | `2.11.0.dev20260119+cu128` | `results/environment.json` |
| CUDA runtime | `12.8` | `results/environment.json` |
| GPU | `NVIDIA GeForce RTX 5070 Ti` | `results/environment.json` |
| GPU memory | `15.92 GB` | `results/environment.json` |
| transformers | `5.9.0` | `results/environment.json` |
| datasets | `4.8.5` | `results/environment.json` |
| peft | `0.19.1` | `results/environment.json` |
| framework | Hugging Face `Trainer` | `src/suite.py` |
| precision | `fp16` | `config/experiment_config.json` |

## 6. Hyperparameters

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| learning rate: Full FT | 0.00002 | `config/experiment_config.json` |
| learning rate: LoRA | 0.0001 | `config/experiment_config.json` |
| learning rate: Adapter | 0.0001 | `config/experiment_config.json` |
| learning rate: IA3 | 0.0005 | `config/experiment_config.json` |
| learning rate: BitFit | 0.0001 | `config/experiment_config.json` |
| batch size | 16 | `config/experiment_config.json` |
| eval batch size | 32 | `config/experiment_config.json` |
| gradient accumulation steps | 4 | `config/experiment_config.json` |
| epoch: Study 1 | 3 | `config/experiment_config.json` |
| epoch: Study 2 | 2 | `config/experiment_config.json` |
| epoch: Study 3 | 5 | `config/experiment_config.json` |
| optimizer | 확인 필요 | 확인 필요 |
| scheduler | 확인 필요 | 확인 필요 |
| max length | 128 | `config/experiment_config.json` |
| LoRA rank | 8 | `config/experiment_config.json` |
| LoRA alpha | 16 | `config/experiment_config.json` |
| LoRA dropout | 0.05 | `config/experiment_config.json` |
| Adapter bottleneck | 64 | `config/experiment_config.json` |
| Adapter dropout | 0.0 | `config/experiment_config.json` |
| weight decay | 0.01 | `config/experiment_config.json` |
| warmup ratio | 0.06 | `config/experiment_config.json` |
| early stopping patience | 2 | `config/experiment_config.json` |

자세한 표는 `results_summary/hyperparameters.md`에 별도로 정리하였다.

## 7. Evaluation Metrics

평가 지표는 `src/suite.py`의 `compute_metrics`에서 확인된다.

| Metric | 의미 | 평가 방식 | 결과 파일 위치 |
|---|---|---|---|
| `accuracy` | 전체 sample 중 정답 비율 | test split prediction | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |
| `macro_f1` | class별 F1의 macro average | test split prediction | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |
| `macro_precision` | class별 precision의 macro average | test split prediction | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |
| `macro_recall` | class별 recall의 macro average | test split prediction | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |
| `train_seconds` | 학습 소요 시간 | run별 측정 | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |
| `trainable_parameter_ratio` | 전체 parameter 중 학습 대상 비율 | model parameter count | `final_metrics.json`, `final_tables/summary_by_task_model_method.csv` |

## 8. Results

| 실험명 | 주요 설정 | Metric | 결과값 | 출처 파일 |
|---|---|---|---:|---|
| 전체 실험 | 3 Studies, 22 task/model, 5 methods, 5 seeds | 완료 run 수 | 550 | 원본 workspace의 `final_tables/all_runs_550.csv`; 업로드 폴더에서는 run-level table 제외 |
| 전체 평균 | Full Fine-tuning | mean Macro-F1 | 0.7588 | `final_tables/summary_by_task_model_method.csv` |
| 전체 평균 | IA3 | mean Macro-F1 | 0.6702 | `final_tables/summary_by_task_model_method.csv` |
| 전체 평균 | LoRA | mean Macro-F1 | 0.6537 | `final_tables/summary_by_task_model_method.csv` |
| 전체 평균 | Adapter | mean Macro-F1 | 0.6506 | `final_tables/summary_by_task_model_method.csv` |
| 전체 평균 | BitFit | mean Macro-F1 | 0.6443 | `final_tables/summary_by_task_model_method.csv` |
| 성능 1위 분포 | Full Fine-tuning | count | 19 | `final_tables/winners_by_metric.csv` |
| 성능 1위 분포 | IA3 | count | 2 | `final_tables/winners_by_metric.csv` |
| 성능 1위 분포 | Adapter | count | 1 | `final_tables/winners_by_metric.csv` |
| 파라미터 효율 1위 분포 | IA3 | count | 22 | `final_tables/winners_by_metric.csv` |
| 안정성 1위 분포 | BitFit | count | 10 | `final_tables/winners_by_metric.csv` |

task/model별 상세 결과는 `results_summary/results_table.md`와 `final_tables/summary_by_task_model_method.csv`를 참조한다.

## 9. Discussion

Full Fine-tuning은 22개 task/model 조합 중 19개에서 Macro-F1 기준 성능 1위로 집계되었다. 이는 Full FT가 강한 기준선임을 보여준다. 그러나 IA3는 `tweet_hate`의 두 model 조건에서 Full FT보다 높은 성능을 보였고, Adapter는 Study 3의 `news_ynat`에서 성능 1위로 집계되었다.

따라서 결과 해석은 다음과 같이 정리할 수 있다.

- 최고 성능이 가장 중요한 경우 Full Fine-tuning을 우선 기준선으로 둔다.
- 일부 작은 hate-speech task에서는 IA3를 대안으로 검토할 수 있다.
- 성능과 효율의 균형을 볼 때 Adapter가 일부 조건에서 경쟁력 있는 선택지로 나타난다.
- LoRA는 parameter-efficient하지만 wall-clock time 절감이 항상 보장되지는 않는다.
- BitFit은 안정성 측면에서 자주 우세하지만, 평균 성능은 Full FT보다 낮다.

학습시간은 Study, dataset 크기, model 조건에 영향을 받으므로 서로 다른 Study 간 절대 비교로 해석하지 않는다.

## 10. Reproducibility

본 저장소는 원본 노트북을 수정하거나 재실행하지 않고 기존 실험 결과를 확인하는 방식을 우선한다.

### 10.1 환경 설치

```powershell
python -m pip install -r requirements.txt
```

실제 실험 환경의 package version은 `results/environment.json`에 기록되어 있다. 완전한 환경 재구성은 CUDA, PyTorch nightly/dev build, GPU driver 상태에 따라 추가 확인이 필요하다.

### 10.2 원본 실험 파일 위치

| 목적 | 파일 |
|---|---|
| 환경 precheck | `notebooks/00_PRECHECK.ipynb` |
| Study 1 실행 notebook | `notebooks/01_STUDY1_BERTWEET_HATE.ipynb` |
| Study 2 실행 notebook | `notebooks/02_STUDY2_MULTITASK.ipynb` |
| Study 3 실행 notebook | `notebooks/03_STUDY3_KOREAN.ipynb` |
| 결과 aggregate notebook | `notebooks/04_AGGREGATE.ipynb` |
| progress monitor notebook | `notebooks/05_PROGRESS_MONITOR.ipynb` |
| 핵심 실행 코드 | `src/suite.py` |
| 실험 설정 | `config/experiment_config.json` |
| 최종 결과 table | `final_tables/*.csv` |

### 10.3 결과 확인 순서

1. `final_tables/summary_by_task_model_method.csv`에서 task/model/method별 평균과 표준편차를 확인한다.
2. `final_tables/winners_by_metric.csv`에서 성능, 파라미터 효율, 안정성 기준 winner를 확인한다.
3. `results_summary/*.md`에서 논문 첨부용 요약 표를 확인한다.
4. run-level 전체 table과 HTML 보고서는 GitHub 업로드 용량과 공개 범위를 줄이기 위해 이 폴더에서 제외하였다.

### 10.4 재현 시 주의사항

- 원본 notebook의 code, output, markdown, metadata를 수정하지 않는다.
- 기존 결과 파일을 덮어쓰지 않는다.
- `SMOKE` 결과는 최종 논문 결론에 사용하지 않는다.
- 재학습 명령은 본 README에서 제공하지 않는다. 실제 재실행 명령은 확인 필요이다.
- checkpoint와 cache는 공개용 경량 repository에 포함하지 않는다.

## 11. Key Insights

최종 결과는 하나의 universal best method보다 목적별 fine-tuning 전략 선택이 더 적절하다는 점을 보여준다. 자세한 해석은 `results_summary/key_insights.md`에 정리하였다.

| 관찰 | 근거 | 출처 파일 |
|---|---|---|
| Full Fine-tuning은 가장 강한 일반 기준선이다. | 성능 1위 19/22, 평균 Macro-F1 0.7588 | `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv` |
| PEFT가 특정 조건에서 Full FT를 넘을 수 있다. | IA3는 `tweet_hate` 2개 model 조건에서 성능 1위, Adapter는 `news_ynat`에서 성능 1위 | `final_tables/winners_by_metric.csv` |
| 파라미터 효율과 실제 학습시간은 같은 문제가 아니다. | IA3가 22/22 조건에서 최저 trainable ratio이나, wall-clock time은 별도 해석 필요 | `final_tables/winners_by_metric.csv`, `final_tables/summary_by_task_model_method.csv` |
| 안정성은 단순 업데이트 전략에서 자주 높게 나타났다. | BitFit이 seed 안정성 1위 10/22 | `final_tables/winners_by_metric.csv` |

## 12. Repository Scope

본 repository는 논문 재현성 확인에 필요한 경량 공개본이다. 실험 재현을 위한 source code, notebook, configuration, environment record, final summary tables를 포함하며, 매 run마다 생성된 checkpoint, optimizer state, raw prediction, trainer history, event log, dataset cache는 포함하지 않는다.

따라서 이 repository는 원본 실험의 모든 중간 산출물을 저장하는 archive가 아니라, 논문 독자가 실험 설계와 결과 요약을 검토하고 같은 설정으로 재실행할 수 있도록 구성한 supplementary repository이다.

## 13. Limitations

- 본 repository는 논문 재현성 확인을 위한 경량 공개본이며, 원본 workspace의 checkpoint, cache, run-level log 전체를 포함하지 않는다.
- Study별 model, dataset, epoch 수가 다르므로 학습시간은 서로 다른 Study 사이에서 절대 비교하기보다 동일 task/model 조건 안에서 해석해야 한다.
- 실험 환경은 `results/environment.json`에 기록된 단일 GPU 환경을 기준으로 하며, 다른 GPU, driver, PyTorch 조합에서는 wall-clock time이 달라질 수 있다.
- 데이터셋별 라이선스와 인용 요구사항은 각 원천 데이터셋 페이지를 기준으로 확인해야 한다.
- optimizer와 scheduler는 Hugging Face `TrainingArguments` 기반으로 실행되었으며, 명시적으로 별도 optimizer/scheduler를 지정한 설정은 `config/experiment_config.json`에서 확인되지 않는다.

## 14. Citation

```bibtex
@misc{project_repository,
  title        = {Fine-Tuning Strategy Reproducibility Repository},
  author       = {LEESY-X},
  year         = {2026},
  howpublished = {\url{https://github.com/LEESY-X/Domain_fine-tuning_experiments}}
}
```

## 15. Collaborators

이 레포지토리는 프로젝트 단위로 관리하며, 필요한 collaborator를 GitHub에서 초대하면 된다.
