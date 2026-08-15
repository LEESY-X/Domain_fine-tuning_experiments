# Hyperparameters Summary

본 문서는 기존 config, source code, 결과 파일에서 확인 가능한 hyperparameter만 정리한다. 확인되지 않는 값은 `확인 필요`로 표시하였다.

## 1. 공통 설정

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| methods | `full_ft`, `lora`, `adapter`, `ia3`, `bitfit` | `config/experiment_config.json` |
| seeds | `42`, `52`, `62`, `72`, `82` | `config/experiment_config.json` |
| max length | 128 | `config/experiment_config.json` |
| batch size | 16 | `config/experiment_config.json` |
| eval batch size | 32 | `config/experiment_config.json` |
| gradient accumulation steps | 4 | `config/experiment_config.json` |
| precision | `fp16` | `config/experiment_config.json` |
| weight decay | 0.01 | `config/experiment_config.json` |
| warmup ratio | 0.06 | `config/experiment_config.json` |
| early stopping patience | 2 | `config/experiment_config.json` |
| dataloader num workers | 2 | `config/experiment_config.json` |
| keep best checkpoint | `true` | `config/experiment_config.json` |
| continue on error | `false` | `config/experiment_config.json` |
| logging strategy | `steps` | `src/suite.py` |
| logging steps | 20 | `src/suite.py` |
| save strategy | `epoch` | `src/suite.py` |
| evaluation strategy | `epoch` | `src/suite.py` |
| metric for best model | `macro_f1` | `src/suite.py` |
| save total limit | 1 | `src/suite.py` |
| report_to | `none` | `src/suite.py` |
| data seed | 42 | `src/suite.py` |
| optimizer | `adamw_torch` | 현재 `config/experiment_config.json`, `src/suite.py` |
| scheduler | `linear` | 현재 `config/experiment_config.json`, `src/suite.py` |
| dropout | method별 설정 참조 | `config/experiment_config.json` |

현재 재현 코드와 후속 실험은 optimizer/scheduler를 명시적으로 고정한다. 다만 공개 패키지에는 원본 550개 run의 `run_config.json`이 없으므로, 원본 집계 생성 당시의 실제 값을 소급해 단정하지 않는다.

## 2. Method별 learning rate

| Method | Learning rate | 출처 파일 |
|---|---:|---|
| Full Fine-tuning | 0.00002 | `config/experiment_config.json` |
| LoRA | 0.0001 | `config/experiment_config.json` |
| Adapter | 0.0001 | `config/experiment_config.json` |
| IA3 | 0.0005 | `config/experiment_config.json` |
| BitFit | 0.0001 | `config/experiment_config.json` |

## 3. LoRA 설정

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| LoRA rank (`r`) | 8 | `config/experiment_config.json` |
| LoRA alpha | 16 | `config/experiment_config.json` |
| LoRA dropout | 0.05 | `config/experiment_config.json` |
| target modules | `query`, `value` | `src/suite.py` |
| modules to save | `classifier` | `src/suite.py` |
| bias | `none` | `src/suite.py` |

## 4. Adapter 설정

| 항목 | 값 | 출처 파일 |
|---|---:|---|
| bottleneck | 64 | `config/experiment_config.json` |
| dropout | 0.0 | `config/experiment_config.json` |
| 삽입 위치 | encoder layer output module | `src/suite.py` |

## 5. IA3 설정

| 항목 | 값 | 출처 파일 |
|---|---|---|
| target modules | `key`, `value`, `intermediate.dense` | `src/suite.py` |
| feedforward modules | `intermediate.dense` | `src/suite.py` |
| modules to save | `classifier` | `src/suite.py` |

## 6. Study별 epoch 및 모델

| Study | Epoch | Models | Tasks | 출처 파일 |
|---|---:|---|---|---|
| Study 1 | 3 | `vinai/bertweet-base` | `measuring_hate_speech` | `config/experiment_config.json` |
| Study 2 | 2 | `vinai/bertweet-base`, `FacebookAI/roberta-base` | `tweet_sentiment`, `finance_sentiment`, `movie_reviews`, `product_reviews`, `tweet_emotion`, `tweet_hate`, `tweet_offensive`, `tweet_irony`, `news_topic` | `config/experiment_config.json` |
| Study 3 | 5 | `klue/roberta-base` | `news_ynat`, `movie_nsmc`, `comment_kmhas_binary` | `config/experiment_config.json` |
