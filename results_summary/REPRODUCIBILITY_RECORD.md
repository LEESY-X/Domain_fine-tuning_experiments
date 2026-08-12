# 논문 재현성 기록

이 문서는 원본 550-run 구성으로부터 보존된 집계 결과와 2026-08-07에 수행한 표준편차 0 후속 실험 100회를 구분하여 재현·감사하는 기록이다. 논문의 Reproducibility Statement, Methods, Appendix, Artifact Availability 문단을 작성할 때 이 문서를 기준으로 삼는다.

## 1. 가장 먼저 확인할 재현성 상태

| 범위 | 현재 증거 수준 | 재현 시 주의점 |
|---|---|---|
| 원본 550-run 구성의 집계 증거 | 110행 method 집계표, 22행 winner 표, 원본 환경 기록, 공개 당시 source/config와 현재 강화 코드가 있음 | 공개 패키지에 원본 run-level metrics·predictions·run config와 exact executed source가 없으므로 집계값의 내부 일관성은 검증할 수 있지만 550회 완료나 실행 당시 설정을 전부 소급 증명할 수는 없음 |
| 표준편차 0 진단 | 원본 집계표의 전체 저장 정밀도로 재계산 가능 | 원본 개별 prediction이 없으므로 원본 붕괴 판정은 aggregate metric fingerprint 기반 진단임 |
| 파일럿 후속 60회 | 60/60 `final_metrics.json`, `predictions.csv`, history, event, status와 aggregate가 있음 | 당시 per-run `trainer_device` 필드는 아직 기록되지 않았고 실험 후 갱신한 환경 파일만 MPS를 표시함 |
| 전체 데이터 후속 40회 | 40/40 per-run 결과와 aggregate, 실제 `trainer_device=mps`, 모델 commit, 데이터 fingerprint가 있음 | Apple MPS 결과이므로 원본 CUDA aggregate와 절대값을 직접 동일 환경 재현으로 간주하면 안 됨 |
| 논문용 파생 표 | 생성 코드, CSV, LaTeX, Markdown, HTML artifact가 있음 | HTML 브라우저 QA는 Chromium 부재로 구조 검증까지만 완료됨. 논문 수치 재현에는 CSV/LaTeX가 기준임 |

### 증거 우선순위

결과가 서로 다르게 보일 때는 다음 순서로 판단한다.

1. 후속 실험 개별 run의 최종 상태와 지표: `final_metrics.json` 및 동일 내용의 최종 `status.json`
2. 개별 예측: `predictions.csv`
3. 학습 과정: `trainer_history.csv`, `epoch_metrics.csv`, `events.jsonl`
4. 후속 집계: `aggregate/all_runs.csv`, `aggregate/summary.csv`
5. 논문용 파생표: `results_summary/paper_followup/*.csv`
6. 원본 550회 결과: `final_tables/*.csv`

`run_config.json`의 `status=RUNNING`은 실행 시작 시점 스냅샷을 의미하며 최종 성공 여부가 아니다. 최종 완료 여부는 `final_metrics.json` 또는 `status.json`에서 확인한다.

## 2. 저장소와 코드 식별자

기록 기준일은 2026-08-07, 기준 branch는 `main`, 기준 Git commit은 `539e3d3588ef3cc4aa6299fec74776bc1afe0d2d`이다. 다만 후속 진단·재학습·보고서 변경은 현재 worktree에 아직 commit되지 않았으므로 논문 제출 전 이 상태를 commit/tag하고 아래 식별자를 갱신해야 한다.

| 파일 또는 실행물 | SHA-256 |
|---|---|
| 후속 결과를 생성할 당시 기록된 `src/suite.py` | `3bcdaa9bafe78748aa3e7658008c984c09c0b0349fcd5fdc55d0d34c2469bdcb` |
| 현재 `src/suite.py` | `d74c8ea79cd2602eed1e12625c29c5ec909b80d51e68a538ba4e1694db3991c4` |
| 현재 `src/result_analysis.py` | `e0cb6be82ea4371cc88a11e80f78974a04f6b5cac852cbcf4c4c99964069414c` |
| 현재 `tools/run_collapse_followup.py` | `06dff620d56c945ed9797981a0d45d5da8d20c1cee8305ad425de05a2bb98656` |
| 현재 `config/experiment_config.json` | `9926e15c400d2f2aa8bd3eb5c9821c0a759e5331fda34aa556cf8ee317cb949a` |
| `config/collapse_followup.json` | `b2c7aef8e00388e9adba4cac898131f58b180bcc535cf36679cef73385863ce1` |
| `requirements-followup.txt` | `e8a20aaca9767e219200dc935cb86c41b0c1f012a2189434f148883bb1f3e517` |

후속 결과를 생성한 suite hash와 현재 suite hash가 다르다. 현재 코드는 결과 저장 필드, MPS 환경 식별, 경로 휴대성, revision 고정과 검증을 보강한 버전이다. 실행 당시 exact source snapshot은 hash만 남고 별도 파일로 보존되지 않았으므로, 현재 코드가 당시 실행 파일과 bit-identical하다고 주장해서는 안 된다. 재제출 전 current code로 대표 조건을 재실행하거나, 결과 생성 commit을 별도 태그로 고정하는 것이 필요하다.

현재 코드는 모든 모드에서 cached `COMPLETE` 결과와 resumable checkpoint의 `run_signature`가 현재 code/config와 일치하는지 확인한다. 또한 strict aggregate는 payload/path identity, `COMPLETE` 상태, run mode, 중복 key, seed 완전성, 전체 550-key 구성을 검증한 뒤에만 CSV를 쓴다. 이 보호는 앞으로의 실행 계보를 강화하지만, 원본 run-level 파일이 없는 과거 550-run 구성을 소급 증명하지는 않는다.

## 3. 연구 범위와 run 수

### 원본 aggregate가 나타내는 550-run 구성

보존된 설계는 모든 task/model 조건에 5개 방법과 5개 seed를 배정한다. 원본 run-level 파일 부재 때문에 이 절의 550은 독립 확인된 완료 수가 아니라 구성 수다.

| Study | 모델 | task 수 | Epoch | 계산식 | Run 수 |
|---|---|---:|---:|---|---:|
| Study 1 | `vinai/bertweet-base` | 1 | 3 | 1 task × 1 model × 5 methods × 5 seeds | 25 |
| Study 2 | `vinai/bertweet-base`, `FacebookAI/roberta-base` | 9 | 2 | 9 tasks × 2 models × 5 methods × 5 seeds | 450 |
| Study 3 | `klue/roberta-base` | 3 | 5 | 3 tasks × 1 model × 5 methods × 5 seeds | 75 |
| 합계 | 3 base models | 13 tasks, 22 task/model 조건 | — | — | 550 |

비교 방법은 `full_ft`, `lora`, `adapter`, `ia3`, `bitfit`이며 seed는 `42`, `52`, `62`, `72`, `82`이다.

### 표준편차 0 후속 실험 100회

| 단계 | 구성 | Run 수 | 용도 |
|---|---|---:|---|
| Pilot 공통 비교 | 4개 문제 조건 × baseline/remedy × 5 seeds | 40 | 원래 붕괴 재현 및 결합 개선안 선별 |
| Pilot Adapter ablation 추가 | Finance/RoBERTa/Adapter × 나머지 4 variants × 5 seeds | 20 | epoch, weighting, learning-rate 조합 비교 |
| 전체 데이터 paired 비교 | 4개 조건 × baseline/remedy × 5 seeds | 40 | 최종 효과 추정 |
| 합계 | — | 100 | 100/100 완료, failure 0 |

후속 네 조건은 다음과 같다.

1. FinancialPhraseBank sentiment / RoBERTa / Adapter
2. FinancialPhraseBank sentiment / RoBERTa / BitFit
3. TweetEval emotion / RoBERTa / BitFit
4. TweetEval emotion / BERTweet / LoRA

## 4. 데이터셋, label 정의와 split 크기

데이터 파일은 저장소에 재배포하지 않으며 실행 시 Hugging Face Datasets 또는 공개 GitHub TSV에서 불러온다.

| Study | Task key | 원천 데이터셋 | Subset/직접 로더 | 클래스 | Train/Validation/Test |
|---|---|---|---|---:|---:|
| 1 | `measuring_hate_speech` | `ucberkeley-dlab/measuring-hate-speech` | source split `train` | 2 | 108444/13556/13556 |
| 2 | `tweet_sentiment` | `cardiffnlp/tweet_eval` | `sentiment` | 3 | 45615/2000/12284 |
| 2 | `finance_sentiment` | `lmassaron/FinancialPhraseBank` | default | 3 | 3872/484/484 |
| 2 | `movie_reviews` | `stanfordnlp/imdb` | default | 2 | 22500/2500/25000 |
| 2 | `product_reviews` | `SetFit/amazon_reviews_multi_en` | default | 5 | 200000/5000/5000 |
| 2 | `tweet_emotion` | `cardiffnlp/tweet_eval` | `emotion` | 4 | 3257/374/1421 |
| 2 | `tweet_hate` | `cardiffnlp/tweet_eval` | `hate` | 2 | 9000/1000/2970 |
| 2 | `tweet_offensive` | `cardiffnlp/tweet_eval` | `offensive` | 2 | 11916/1324/860 |
| 2 | `tweet_irony` | `cardiffnlp/tweet_eval` | `irony` | 2 | 2862/955/784 |
| 2 | `news_topic` | `fancyzhx/ag_news` | default | 4 | 108000/12000/7600 |
| 3 | `news_ynat` | `klue` | `ynat` | 7 | 41110/4568/9107 |
| 3 | `movie_nsmc` | `e9t/nsmc` 공개 TSV | `direct=nsmc` | 2 | 134995/15000/49997 |
| 3 | `comment_kmhas_binary` | `adlnlp/K-MHaS` 공개 TSV | `direct=kmhas` | 2 | 78977/8776/21939 |

### 전처리와 split 규칙

- 공통 컬럼은 `sample_id`, `text`, `labels`로 표준화한다.
- text 또는 label이 없는 행과 정의된 label 범위를 벗어난 행은 제외한다.
- `measuring_hate_speech`는 연속형 `hate_speech_score`가 아니라 ordinal `hatespeech` 필드를 사용하며, 값이 `>=1`이면 1, 그 미만이면 0으로 이진화한다.
- `comment_kmhas_binary`는 label 목록이 정확히 `[8]`이면 0, 그 외는 1로 변환한다.
- 원본에 train/validation/test가 모두 있으면 그대로 사용한다.
- test만 있으면 train의 10%를 validation으로 분리하고 원본 test를 유지한다.
- validation만 있으면 train의 10%를 새 validation으로 분리하고 원본 validation을 test로 사용한다.
- 단일 split만 있으면 80% train, 나머지 20%를 validation/test로 절반씩 분리한다.
- split과 제한 샘플 선택 seed는 42이며 가능한 경우 stratified split을 사용한다. class 수 부족으로 stratification이 실패하면 동일 seed의 비층화 split로 fallback한다.
- Pilot 제한은 train 1,024, validation 512, test 2,048이지만 원래 split이 더 작으면 원래 크기를 유지한다.
- tokenization은 truncation과 `max_length=128`을 사용한다. BERTweet은 slow tokenizer, 나머지는 fast tokenizer를 요청하고 batch 안에서 dynamic padding을 사용한다.

### 후속 실험 데이터 fingerprint

| 범위 | Task | Rows train/val/test | Train fingerprint | Validation fingerprint | Test fingerprint |
|---|---|---:|---|---|---|
| Pilot | Finance | 1024/484/484 | `1a24caecca529dc6` | `41f3cb00fc2218e7` | `10787703ae1cbb98` |
| Pilot | Emotion | 1024/374/1421 | `f9d462a9feb1b9c9` | `60b428ec5c3a181c` | `03c09f8a7d7ab5a9` |
| Full | Finance | 3872/484/484 | `6b104ca5c71edca8` | `41f3cb00fc2218e7` | `10787703ae1cbb98` |
| Full | Emotion | 3257/374/1421 | `c651013542e077f3` | `60b428ec5c3a181c` | `03c09f8a7d7ab5a9` |

Fingerprint와 행 수는 `results/followup/provenance.json`에 저장되어 있다. 이 값은 Hugging Face Dataset fingerprint이며 raw-example cryptographic checksum은 아니다. 데이터셋 repository revision 자체는 고정되어 있지 않으므로 fingerprint가 다르면 동일 데이터로 간주하지 말고 원천 revision을 조사해야 한다.

## 5. 모델과 fine-tuning 방법

### Base model revision

| 모델 | 후속 실험 commit |
|---|---|
| `FacebookAI/roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| `vinai/bertweet-base` | `b349c1243407b0dcffeabb2337497477286e27ab` |
| `klue/roberta-base` | 현재 미고정; 원본 Study 3 재현 전 commit 고정 필요 |

### 방법 구현

| 방법 | 구현 |
|---|---|
| Full fine-tuning | 전체 파라미터 학습 |
| LoRA | `r=8`, `alpha=16`, dropout 0.05, `query`와 `value` target, classifier 저장, bias `none` |
| Adapter | 각 encoder layer output에 bottleneck 64 residual adapter 삽입, adapter dropout 0.0, 원래 backbone 동결, classifier 학습 |
| IA3 | `key`, `value`, `intermediate.dense` target, feed-forward target은 `intermediate.dense`, classifier 저장 |
| BitFit | backbone을 동결하고 이름이 `.bias`로 끝나는 bias 및 classifier/score head만 학습 |

모델은 sequence classification head로 불러오며 `ignore_mismatched_sizes=True`, attention implementation은 `eager`다. 분류 head가 새로 초기화되는 경우 `set_seed(seed)` 이후 생성되므로 model initialization도 run seed의 통제를 받는다.

## 6. 현재 후속·재구성 코드의 공통 학습 설정

| 항목 | 값 |
|---|---|
| Seeds | 42, 52, 62, 72, 82 |
| Data seed | 42 |
| Max sequence length | 128 |
| Per-device train batch | 16 |
| Per-device eval batch | 32 |
| Gradient accumulation | 4 |
| 명목상 effective train batch | 64 examples/device/update |
| Optimizer (현재 후속·재구성 코드) | `adamw_torch` |
| LR scheduler (현재 후속·재구성 코드) | linear |
| Weight decay | 0.01 |
| Warmup ratio | 0.06 |
| Evaluation/save | epoch마다 |
| Best model 기준 | validation Macro-F1, greater is better |
| Early stopping patience | 2 evaluation rounds |
| Checkpoint retention | 원본 설정은 best checkpoint 유지; 후속은 저장공간을 위해 학습 후 checkpoint 삭제 |
| Logging | 20 steps마다, 외부 tracker 비활성화 (`report_to=none`) |

원본 공개 source/config에는 optimizer와 scheduler 이름이 명시되지 않았고 per-run config도 없다. 따라서 위 두 값은 현재 후속·재구성 코드의 설정이며 원본 실행값으로 단정하지 않는다. 원본 설정의 `precision=fp16`은 CUDA가 사용 가능할 때만 Trainer의 fp16 flag를 켠다. Apple MPS 후속 실험에서는 설정 파일의 표기가 `fp16`이어도 코드상 fp16 Trainer flag가 활성화되지 않았다.

Method별 기본 learning rate는 Full FT `2e-5`, LoRA `1e-4`, Adapter `1e-4`, IA3 `5e-4`, BitFit `1e-4`다.

`train_seconds`는 `trainer.train()` 호출 직전부터 완료 직후까지의 wall-clock 시간이다. 모델·데이터 로딩과 최종 test prediction 시간은 포함하지 않는다.

## 7. 표준편차 0 후속 설계

### Variant 정의

| Variant | Epoch | Learning rate | Loss weighting |
|---|---:|---|---|
| `baseline` | 2 | 원래 method LR | 없음 |
| `longer` | 최대 5 | 원래 method LR | 없음 |
| `weighted` | 2 | 원래 method LR | inverse-sqrt frequency |
| `longer_weighted` | 최대 5 | 원래 method LR | inverse-sqrt frequency |
| `higher_lr` | 2 | Adapter/BitFit `5e-4`, LoRA `3e-4` | 없음 |
| `higher_lr_weighted` | 최대 5 | Adapter/BitFit `5e-4`, LoRA `3e-4` | inverse-sqrt frequency |

Class `c`의 train count를 `n_c`라 할 때 개선 조건의 weight는 `n_c^(-1/2)`로 계산하고 class 간 평균이 1이 되도록 정규화한 뒤 cross-entropy에 적용한다. 모든 weight는 train split만 사용해 계산한다.

전체 데이터 최종 비교의 remedy는 `higher_lr_weighted`다. 이 조건은 epoch, learning rate, class weighting을 동시에 바꾸므로 개별 요소의 독립 인과효과를 추정하는 설계가 아니다.

## 8. 지표와 통계 정의

### 성능과 효율

- Accuracy: 전체 test sample 중 정답 비율
- Macro precision/recall/F1: class별 값을 동일 가중 평균하며 zero division은 0으로 처리
- Trainable parameter ratio: trainable parameters / total parameters
- Run별 test metric은 validation Macro-F1 기준 best checkpoint를 복원한 뒤 test split에서 한 번 계산
- 집계 평균은 5개 seed 산술평균, SD는 `ddof=1`인 표본 표준편차

### 예측 붕괴 진단

- Constant-class collapse: test prediction에서 등장한 class 수가 정확히 1
- Near-collapse: 최빈 예측 class 비율이 0.98 이상
- Predicted-class coverage: 예측한 class 수 / 전체 class 수
- Normalized prediction entropy: 예측 class 분포 entropy를 `log(K)`로 나눈 값
- Threshold 오해를 피하기 위해 `majority_prediction_rate` 연속값과 class별 prediction count도 함께 저장

K-class 문제에서 모든 test sample을 하나의 class로 예측하고 그 class의 실제 비율이 `p`라면 `accuracy=p`, `macro precision=p/K`, `macro recall=1/K`, `macro F1=(2p/(1+p))/K`다. 원본 exact-zero 네 행은 이 관계를 절대오차 `1e-12` 이내에서 만족한다.

### Paired 추론

- 동일 seed의 `remedy F1 - baseline F1`을 paired difference로 사용
- 평균 차이와 표본 SD를 계산
- 95% CI는 자유도 4의 paired t interval
- p-value는 5개 차이의 부호를 모두 뒤집는 `2^5=32`개 조합을 열거한 양측 exact sign-flip test
- 다섯 차이가 모두 같은 방향일 때도 최소 양측 p-value는 `0.0625`
- 네 조건에 대한 multiplicity adjustment는 적용하지 않았으며 분석은 탐색적임

따라서 paired t CI가 0을 포함하지 않더라도 exact test 기준 `p<0.05`라고 쓰지 않는다.

## 9. 실행 환경

### 원본 550회 환경 기록

| 항목 | 값 |
|---|---|
| 기록 시각 | 2026-06-24 19:24:34 KST |
| Python | 3.10.19, Anaconda, Windows/MSVC build |
| PyTorch | `2.11.0.dev20260119+cu128` |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 약 15.92 GB |
| Transformers | 5.9.0 |
| Datasets | 4.8.5 |
| PEFT | 0.19.1 |

### 후속 전체 데이터 환경

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-07 |
| Python | 3.13.7, arm64 |
| PyTorch | 2.13.0 |
| Accelerator | Apple MPS |
| CUDA | 사용 안 함 |
| Transformers | 5.9.0 |
| Datasets | 4.8.5 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| NumPy | 2.5.1 |
| sentencepiece | 0.2.2 |

Pilot의 experiment-level 환경 파일도 같은 Python/package/MPS 상태를 기록하지만 실행 종료 후 22:13 KST에 갱신된 기록이다. Pilot per-run 파일에는 `trainer_device`가 없고 구버전 runtime detector가 `CPU`로 기록했으므로, Pilot 장치를 per-run 증거로 확정해서 인용하지 않는다. 전체 데이터 40개 run은 `final_metrics.json`에 `trainer_device=mps`를 직접 기록한다.

## 10. 목표 재현값

| 조건 | Baseline Macro-F1 | Remedy Macro-F1 | Paired ΔF1 [95% CI] | Collapse baseline→remedy | Exact p |
|---|---:|---:|---:|---:|---:|
| Finance/RoBERTa/Adapter | 0.2759 ± 0.0338 | 0.8486 ± 0.0087 | +0.5727 [0.5239, 0.6215] | 2/5→0/5 | 0.0625 |
| Finance/RoBERTa/BitFit | 0.2482 ± 0.0000 | 0.8348 ± 0.0085 | +0.5866 [0.5760, 0.5972] | 5/5→0/5 | 0.0625 |
| Emotion/RoBERTa/BitFit | 0.1410 ± 0.0000 | 0.7646 ± 0.0056 | +0.6236 [0.6167, 0.6306] | 5/5→0/5 | 0.0625 |
| Emotion/BERTweet/LoRA | 0.1410 ± 0.0000 | 0.7840 ± 0.0037 | +0.6430 [0.6384, 0.6475] | 5/5→0/5 | 0.0625 |

전체 데이터 remedy는 20/20 seed에서 baseline보다 높았고, 20/20 run에서 constant-class collapse가 없었으며, 20/20 run이 전체 class를 예측했다. 평균 학습시간은 baseline의 2.32–2.56배였다.

## 11. 재실행 절차

명령은 저장소 root에서 실행한다. `python` 별칭이 없는 환경이 있으므로 `python3`를 사용한다.

### 후속 실험 환경 구성

```bash
python3 -m venv .venv-followup
source .venv-followup/bin/activate
python3 -m pip install -r requirements-followup.txt
```

모델·데이터 다운로드가 필요하므로 네트워크와 각 원천 데이터 접근 권한이 필요하다. macOS에서는 MPS 지원 PyTorch build, CUDA 재현에서는 호환되는 CUDA/PyTorch build를 사용한다.

### 원본 aggregate의 exact-zero 진단

```bash
python3 tools/analyze_result_variance.py
```

### 실제 수행한 60-run Pilot 구성 재현

```bash
python3 tools/run_collapse_followup.py \
  --variant baseline \
  --variant higher_lr_weighted

python3 tools/run_collapse_followup.py \
  --condition finance_roberta_adapter \
  --variant longer \
  --variant weighted \
  --variant longer_weighted \
  --variant higher_lr
```

인자 없이 runner를 실행하면 현재 config의 6 variants를 네 조건 모두에 적용해 120개 job을 만든다. 이는 보존된 60-run Pilot과 다른 설계이므로 논문 재현 명령으로 사용하지 않는다.

### 실제 수행한 40-run 전체 데이터 paired 비교 재현

```bash
python3 tools/run_collapse_followup.py \
  --full-data \
  --variant baseline \
  --variant higher_lr_weighted
```

CUDA만 허용하려면 `--require-cuda`를 추가한다. 기존 결과와 다른 code/config signature로 같은 experiment id를 재사용하면 stale-result 오류가 발생하므로, 새 experiment id를 쓰거나 의도적인 재실행일 때만 `--force`를 사용한다.

### 집계 정리와 논문 표 재생성

```bash
python3 tools/sanitize_followup_artifacts.py
python3 tools/build_paper_followup_report.py
```

Sanitizer는 machine-local 절대경로를 제거하고 aggregate를 다시 만든다. 논문 표 generator는 원본 4행, Pilot 60행, Full 40행과 paired seed 완전성을 확인한 후 CSV·LaTeX·Markdown·canonical artifact를 생성한다.

### 코드와 결과 검증

```bash
python3 tools/create_colab_a100_notebook.py
python3 tools/verify_colab_parity.py
python3 tools/test_result_analysis.py
python3 tools/test_followup_plan.py
python3 tools/test_recovery.py
python3 tools/validate_suite.py
python3 -m compileall -q src tools
git diff --check
```

`tools/test_recovery.py`는 PyTorch가 설치된 호환 환경에서 실행한다.

### 현재 코드의 550-job 재구성 실행 경로

- `colab_a100_full_550_runs.ipynb`: 전체 550회 Colab 실행
- `colab_a100_low_drive_550_runs.ipynb`: Drive 사용량을 줄인 실행
- 로컬 notebook: `notebooks/01_*`부터 `04_aggregate.ipynb`까지 순서대로 실행

현재 코드의 재구성 실행은 `run_study(study, run_mode="PAPER")`를 사용하며 CUDA precheck를 요구한다. 세 Study를 모두 실행한 뒤 strict integrity 검사를 포함한 `aggregate("PAPER")`를 수행한다. 실행 당시 source snapshot, revisions, dependency lock이 없으므로 bit-identical 원본 재실행으로 부르지 않는다.

## 12. 보존된 run-level 산출물

각 후속 run directory에는 다음 파일이 있다.

| 파일 | 역할 |
|---|---|
| `run_config.json` | 실행 시작 시점의 조건, 모델 commit, trainable parameters, class count |
| `status.json` | 현재/최종 실행 상태 |
| `events.jsonl` | 시작·완료·실패 event log |
| `epoch_metrics.csv` | callback 기반 step/checkpoint 기록 |
| `trainer_history.csv` | Trainer log history와 validation metric |
| `predictions.csv` | `sample_id`, true label, predicted label |
| `final_metrics.json` | 최종 test metric, 진단, timing, best checkpoint 이름 |

Pilot은 각 파일 60개, Full은 각 파일 40개이며 모든 `final_metrics.json`의 상태가 `COMPLETE`다. 모델 checkpoint와 dataset cache는 저장소 크기와 배포 문제 때문에 제외했다.

## 13. 핵심 산출물 checksum

| 산출물 | SHA-256 |
|---|---|
| `final_tables/summary_by_task_model_method.csv` | `bcf303162463a4ec3c706269c10f50a9d995355389997611c220d36211f4475c` |
| `final_tables/winners_by_metric.csv` | `d437c9750417c4018f0ecde906debc029ea9af6d3ba4579256ff527e8389e228` |
| Pilot `aggregate/all_runs.csv` | `dc217ff4d30b5304197c2f8c8e862d114840bdbac51aaeed8fdcf6e9c333be4c` |
| Pilot `aggregate/summary.csv` | `ac6c869cd48a013bf3b845647630bd22c3c66eaa80ab5f3aa8b090b8a5f57c64` |
| Full `aggregate/all_runs.csv` | `594d834d41232f212b78596d5a37390b87d611cc51e381070d8f2b43f56782fa` |
| Full `aggregate/summary.csv` | `8997660cf18fac8b3e976357a76423b9dfce67893dc4ab17fc332e96952af1ab` |
| `paper_followup/table_3_full_comparison.csv` | `99a4d5e835df8dc0554e4dc2b582888a22d9bf90af40116e9558ea3f3e7ea6f9` |
| `paper_followup/appendix_full_per_seed.csv` | `63dd58beeeed93ca08733e43365bc81cc6d1161b0fdc80e4bd58cd2a978df1f2` |
| `paper_followup/paper_tables.tex` | `481263e0d87c66d13aff67da588fb1dbb45eb25ee9d0d10d2ccb2ef110d18e34` |

파일을 의도적으로 재생성하면 serialization이나 timestamp 변경으로 checksum이 바뀔 수 있다. 값이 바뀐 경우 run count, seed coverage와 목표 재현값을 다시 검증하고 이 표를 갱신한다.

## 14. 현재 검증 결과

- `tools/validate_suite.py`: PASS, Study run 설계 25/450/75 확인
- `tools/verify_colab_parity.py`: PASS, 두 Colab notebook과 local suite/config 동일
- `tools/test_result_analysis.py`: PASS
- `tools/test_followup_plan.py`: PASS
- `tools/test_recovery.py`: 현재 감사 환경에서는 미검증 (`ModuleNotFoundError: No module named 'torch'`); PyTorch 호환 환경의 실행 로그가 확보된 경우에만 PASS로 갱신
- Pilot: 60 runs, 12 aggregate groups, seed 누락·중복 없음
- Full: 40 runs, 8 aggregate groups, seed 누락·중복 없음
- 원본 summary: 110행, exact-zero F1 SD 4행
- JSON syntax, Python compile, `git diff --check`: PASS
- follow-up/report artifact 내 machine-local 절대경로: 없음
- 10 MB 초과 follow-up/report 파일: 없음
- HTML package validation: PASS
- HTML browser verification: `structural_only`; 설치된 Chromium이 없어 interactive source dialog와 viewport QA는 미검증

## 15. 논문에서 반드시 공개하거나 명시할 항목

1. 5개 seed와 data split seed 42
2. Study별 task/model/epoch와 전체 run 수 계산
3. 데이터 원천, subset, label 변환, split 규칙과 split 크기
4. 모델 이름·revision과 모든 PEFT 설정
5. optimizer, scheduler, learning rate, batch, accumulation, max length, warmup, weight decay, early stopping
6. Macro metric 정의, sample SD, collapse/near-collapse 정의
7. 후속 baseline과 remedy가 바꾼 요소 세 가지
8. paired 비교와 exact test 방식, n=5의 p-value 해석 한계
9. 원본 CUDA와 후속 MPS의 환경 차이
10. 원본 run-level evidence 부재와 후속 run-level evidence 보존 범위
11. 코드·설정·결과 checksum 또는 DOI/tag/commit
12. dataset/model license와 사용 조건

## 16. 남아 있는 재현성 격차와 제출 전 조치

| 격차 | 논문 영향 | 권장 조치 |
|---|---|---|
| 현재 변경이 commit되지 않음 | 독자가 동일 code state를 checkout할 수 없음 | 제출 버전을 commit하고 release tag 및 commit SHA를 본 문서에 기록 |
| 실행 당시 suite source는 hash만 있고 snapshot 없음 | 후속 결과와 현재 코드의 bit-level 연결이 불완전 | 현재 코드로 대표 조건 재실행 또는 실행 source snapshot을 release에 포함 |
| 원본 550회 run-level 결과 미포함 | 원본 붕괴 및 설정을 per-run으로 직접 감사할 수 없음 | 가능하면 anonymized all-runs table과 run config 공개, 불가하면 부재를 명시 |
| 데이터셋 revision 미고정 | 원천 dataset 변경 시 split fingerprint가 달라질 수 있음 | 각 dataset revision/commit을 고정하고 full split fingerprint 생성 |
| `klue/roberta-base` revision 미고정 | Study 3 model drift 가능 | revision을 config에 추가 |
| 원본 optimizer/scheduler per-run 증거 없음 | 현재 명시값을 과거 사실로 단정할 수 없음 | 현재 재현 코드의 설정이라고 표현하고 원본 원시 log 확보 시 갱신 |
| 원본 dependency가 최소 버전 위주 | 과거 환경 복원이 약함 | 원본 CUDA 환경용 exact lock/conda export 추가 |
| OS·driver·MPS 세부 버전 미기록 | 장치별 성능·결정성 차이 해석 제한 | OS build, CUDA driver/cuDNN 또는 macOS/MPS 정보를 다음 실행부터 저장 |
| checkpoint 미보존 | exact model weight 재검증 불가 | 대표 run checkpoint 또는 model state checksum을 선택적으로 보존 |
| CUDA와 MPS 혼재 | cross-environment 절대값 비교 제한 | 핵심 네 조건을 동일 CUDA 환경에서 재실행 |
| 5 seeds | exact 양측 검정 최소 p=0.0625 | confirmatory 실험은 사전등록 후 seed 수 확대 |
| 전체 데이터 factorial ablation 없음 | remedy 구성요소별 효과 분리 불가 | LR × weighting × epoch 전체 factorial 실험 수행 |
| Dataset license/citation 미정리 | 배포·논문 인용 불완전 | 각 원천의 license, canonical paper, URL을 최종 참고문헌과 artifact 문서에 추가 |

## 17. 기준 파일 목록

- 전체 protocol: `config/experiment_config.json`
- 후속 설계: `config/collapse_followup.json`
- 학습 구현: `src/suite.py`
- 붕괴·paired 통계: `src/result_analysis.py`
- 후속 실행: `tools/run_collapse_followup.py`
- 표 생성: `tools/build_paper_followup_report.py`
- pinned top-level dependency versions: `requirements-followup.txt` (complete transitive/OS lock 아님)
- 모델 commit·데이터 fingerprint: `results/followup/provenance.json`
- 원본 환경: `results/environment.json`
- 후속 환경: `results/followup/collapse_followup_v2_full/environment.json`
- 원본 aggregate: `final_tables/summary_by_task_model_method.csv`, `final_tables/winners_by_metric.csv`
- 후속 aggregate: `results/followup/collapse_followup_v2_{pilot,full}/aggregate/`
- 논문용 결과: `results_summary/paper_followup/`
