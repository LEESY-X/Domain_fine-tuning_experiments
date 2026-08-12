# Seed Variance and Prediction-Collapse Audit

## 결론

- 공개 요약표 110개 조건 중 `f1_sd=0`은 4개다.
- 그중 단일-class 예측의 metric fingerprint를 정확히 만족하는 조건은 4개다.
- CSV에는 원래 부동소수점 정밀도가 저장되어 있으므로 표시 반올림으로 0이 된 것이 아니다.
- 이 네 행은 단일-class prediction과 일치하는 aggregate fingerprint이며, 원본 prediction 부재 때문에 과거 run의 collapse를 직접 확인한 것은 아니다.

## 붕괴 조건

| Task | Model | Method | Seeds | Macro-F1 | F1 SD | Classes | 판정 |
|---|---|---|---:|---:|---:|---:|---|
| finance_sentiment | FacebookAI/roberta-base | bitfit | 5 | 0.248162559447 | 0.000000000000 | 3 | fingerprint-consistent with constant-class collapse |
| finance_sentiment | FacebookAI/roberta-base | adapter | 5 | 0.248162559447 | 0.000000000000 | 3 | fingerprint-consistent with constant-class collapse |
| tweet_emotion | FacebookAI/roberta-base | bitfit | 5 | 0.140980293077 | 0.000000000000 | 4 | fingerprint-consistent with constant-class collapse |
| tweet_emotion | vinai/bertweet-base | lora | 5 | 0.140980293077 | 0.000000000000 | 4 | fingerprint-consistent with constant-class collapse |

## 판정 원리

K-class 분류기가 모든 표본을 하나의 class로 예측하고 그 class의 실제 비율을 p라고 하면 `accuracy=p`, `macro_precision=p/K`, `macro_recall=1/K`, `macro_f1=(2p/(1+p))/K`가 된다. 위 조건들은 저장된 네 metric이 이 관계를 1e-12 절대오차 이내에서 동시에 만족한다.

## 후속 실험

원래 2-epoch 조건을 기준선으로 유지하고, 학습 epoch 증가, class-weighted loss, 상향 learning rate를 분리한 뒤 결합 조건까지 비교한다. 각 run에는 예측 class 수, 최대 class 예측률, 예측 entropy, label count를 저장해 같은 문제가 다시 숨지 않도록 한다.

## 붕괴 행 제외 stability sensitivity analysis

| Method | Lowest-SD conditions after exclusion |
|---|---:|
| bitfit | 8 |
| ia3 | 5 |
| full_ft | 3 |
| lora | 3 |
| adapter | 3 |

이 순위도 원본의 나머지 run-level prediction을 직접 검사한 결과는 아니므로 sensitivity analysis로만 해석한다.
