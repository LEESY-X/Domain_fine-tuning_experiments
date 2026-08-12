# 표준편차 0 진단 및 붕괴 후속 재학습: 논문용 결과

## 결론

원본 결과의 Macro-F1 표준편차 0은 소수점 잘림이 아니다. 전체 정밀도 aggregate에서 네 행 모두 단일 클래스 예측의 accuracy·macro precision·macro recall·macro F1 관계와 정확히 일치한다. 따라서 이 네 행은 **constant-class collapse와 일치하는 aggregate 수준의 퇴화 안정성 증거**로 해석한다. 원본 run-level predictions가 없어 각 과거 run의 실제 예측을 직접 확인한 것은 아니다.

## Table 1. 원본 exact-zero SD 진단

| Condition | Macro-F1 mean ± SD | Collapse fingerprint | Interpretation |
|---|---:|---:|---:|
| FinancialPhraseBank sentiment / FacebookAI/roberta-base / BitFit | 0.2482 ± 0.0000 | Yes | Aggregate consistent with constant-class collapse |
| FinancialPhraseBank sentiment / FacebookAI/roberta-base / Adapter | 0.2482 ± 0.0000 | Yes | Aggregate consistent with constant-class collapse |
| TweetEval emotion / FacebookAI/roberta-base / BitFit | 0.1410 ± 0.0000 | Yes | Aggregate consistent with constant-class collapse |
| TweetEval emotion / vinai/bertweet-base / LoRA | 0.1410 ± 0.0000 | Yes | Aggregate consistent with constant-class collapse |

## Table 2. 파일럿 ablation

FinancialPhraseBank/RoBERTa/Adapter, train=1,024. 표의 SD는 5개 seed의 표본 표준편차다.

| Variant | Macro-F1 mean ± SD | Collapse | Full-class coverage |
|---|---:|---:|---:|
| Baseline (2 ep.) | 0.2482 ± 0.0000 | 5/5 | 0/5 |
| Longer (max 5 ep.) | 0.2482 ± 0.0000 | 5/5 | 0/5 |
| Weighted (2 ep.) | 0.2482 ± 0.0000 | 5/5 | 0/5 |
| Longer + weighted (max 5 ep.) | 0.2482 ± 0.0000 | 5/5 | 0/5 |
| Higher LR (2 ep.) | 0.2482 ± 0.0000 | 5/5 | 0/5 |
| Higher LR + weighted (max 5 ep.) | 0.8040 ± 0.0063 | 0/5 | 5/5 |

## Table 3. 전체 데이터 paired 재학습

개선 조건은 early stopping을 포함한 최대 5 epoch + 높은 method-specific learning rate + inverse-square-root-frequency class weighting을 결합했다. 동일 seed끼리 paired 비교했다.
Class weight는 train label count를 n_c라 할 때 n_c^(-1/2)를 계산한 뒤 class 간 평균이 1이 되도록 정규화하여 cross-entropy에 적용했다.
Constant-class collapse는 테스트 예측 클래스 수가 1인 run, near-collapse는 최빈 예측 비율이 0.98 이상인 run으로 정의했다. 다만 threshold 주변 정보 손실을 피하기 위해 표와 appendix에는 최빈 예측 비율의 연속값도 함께 보존했다.
95% CI는 동일 seed의 Macro-F1 차이에 대한 paired t interval이며, p-value는 2-sided exact sign-flip test로 계산했다.

| Condition | Train/Test n | Baseline F1 | Remedy F1 | Paired ΔF1 [95% CI] | Base collapse | Remedy collapse | Base majority rate | Remedy majority rate | Improved seeds | Remedy full coverage | Train-time × | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinancialPhraseBank sentiment / FacebookAI/roberta-base / Adapter | 3872/484 | 0.2759 ± 0.0338 | 0.8486 ± 0.0087 | +0.5727 [+0.5239, +0.6215] | 2/5 | 0/5 | 0.976 | 0.546 | 5/5 | 5/5 | 2.34× | 0.0625 |
| FinancialPhraseBank sentiment / FacebookAI/roberta-base / BitFit | 3872/484 | 0.2482 ± 0.0000 | 0.8348 ± 0.0085 | +0.5866 [+0.5760, +0.5972] | 5/5 | 0/5 | 1.000 | 0.552 | 5/5 | 5/5 | 2.32× | 0.0625 |
| TweetEval emotion / FacebookAI/roberta-base / BitFit | 3257/1421 | 0.1410 ± 0.0000 | 0.7646 ± 0.0056 | +0.6236 [+0.6167, +0.6306] | 5/5 | 0/5 | 1.000 | 0.387 | 5/5 | 5/5 | 2.56× | 0.0625 |
| TweetEval emotion / vinai/bertweet-base / LoRA | 3257/1421 | 0.1410 ± 0.0000 | 0.7840 ± 0.0037 | +0.6430 [+0.6384, +0.6475] | 5/5 | 0/5 | 1.000 | 0.381 | 5/5 | 5/5 | 2.52× | 0.0625 |

## 논문에서 명확히 말할 수 있는 사실

1. 원본 110개 집계 행 중 4개 행의 Macro-F1 SD는 저장된 전체 정밀도에서도 정확히 0이었다.
2. 네 행 모두 단일 클래스 예측의 metric fingerprint와 일치하므로, 해당 0은 반올림 결과가 아니며 seed 전반의 constant-class collapse와 일치하는 aggregate 증거다.
3. 1,024-example 후속 baseline 20/20 run은 constant-class collapse를 직접 보였다. 이는 제한 데이터 후속 관찰이며 원본 실행의 동일 재현은 아니다.
4. 파일럿의 결합 개선 조건에서는 20/20 run에서 constant-class collapse가 사라졌지만, BERTweet-LoRA emotion 조건은 일부 seed가 모든 클래스를 예측하지 않았고 SD도 상대적으로 컸다.
5. BERTweet-LoRA emotion의 파일럿 불안정성은 전체 데이터에서 다시 관찰되지 않았다. 전체 데이터 개선 조건은 `0.7840 ± 0.0037`, full-class coverage 5/5였다. 파일럿은 사후 탐색적 비교이며 최종 효과 추정치가 아니다.
6. 전체 데이터 paired 비교에서 붕괴 run 수가 줄고 평균 Macro-F1이 증가한 조건은 4/4개였다. 세부 효과 크기와 seed 일관성은 Table 3에 제시했다.
7. n=5에서는 모든 paired 차이가 같은 방향이어도 exact two-sided sign-flip p의 최소값이 0.0625이므로, 이번 후속 결과만으로 p<0.05의 통계적 유의성을 주장할 수 없다.
8. 원본은 CUDA/RTX 환경, 후속은 Apple MPS 환경이므로 원본 aggregate와 현재 baseline의 차이는 cross-environment replication 차이를 포함한다. 개선 조건의 효과는 후속 환경 내부 paired 비교로 해석해야 한다.
9. 네 조건의 p-value는 탐색적이며 multiple-comparison adjustment를 적용하지 않았다.
10. 결합 개선 조건의 평균 학습시간은 기준선의 2.32–2.56배로, 붕괴 완화에는 명확한 계산 비용 증가가 동반됐다.

## 논문에 그대로 사용할 수 있는 한국어 문장

“초기 집계 결과에서 네 개의 task–model–method 조합은 Macro-F1 표준편차가 정확히 0이었다. 전체 정밀도 지표를 재검토한 결과, 이 값들은 단일 클래스 예측에서 유도되는 macro precision, recall, F1 관계와 정확히 일치하였다. 따라서 네 행은 반올림에 의한 0이나 바람직한 seed 안정성보다 constant-class prediction과 일치하는 aggregate 증거로 해석한다. 원본 run-level prediction이 없으므로 모든 과거 seed의 실제 예측을 직접 확인한 것은 아니다.”

“후속 실험에서는 예측 클래스 수, 최빈 예측 비율, normalized prediction entropy를 run별로 저장하였다. 또한 동일한 5개 seed에서 원본 2-epoch 설정과 early stopping을 포함한 최대 5 epochs, 상향 learning rate, inverse-square-root-frequency class weighting을 결합한 설정을 paired 비교하였다. Table 3은 평균 Macro-F1의 변화와 함께 붕괴 run 수를 보고하므로, 성능 안정성과 퇴화 안정성을 구분한다.”

## English paper-ready wording

“Four task–model–method combinations in the original aggregate exhibited an exactly zero standard deviation of Macro-F1 across five reported seeds. Re-evaluation at full stored precision showed that their accuracy, macro-precision, macro-recall, and Macro-F1 jointly matched the analytical fingerprint of constant-class prediction. We therefore treat these rows as aggregate evidence consistent with degenerate constant-class stability, rather than as direct proof of the historical predictions.”

“The follow-up recorded the number of predicted classes, majority-prediction rate, and normalized prediction entropy for every run. We paired the original two-epoch configuration with a combined remedy using up to five epochs with early stopping, a higher method-specific learning rate, and inverse-square-root-frequency class weighting under the same five seeds. Because five pairs permit a minimum two-sided exact sign-flip p-value of 0.0625, we report effect sizes, seed-wise direction, and collapse counts without claiming conventional statistical significance.”

## 주장하면 안 되는 내용

- “표준편차가 0이므로 가장 안정적이다.” 붕괴 안정성을 정상 안정성으로 오해한다.
- “class weighting 하나가 붕괴의 원인이자 해결책이다.” 전체 데이터 개선 조건은 세 변경을 결합했으며 개별 인과 효과가 분리되지 않았다.
- “모든 PEFT와 모든 task에 일반화된다.” 후속 범위는 원본에서 문제가 발견된 네 조건뿐이다.
- “통계적으로 유의하다(p<0.05).” n=5 exact paired test로는 그 결론에 도달할 수 없다.

## 재현 정보

- Seeds: `[42, 52, 62, 72, 82]`
- Full-data Trainer devices: `['mps']`
- Pinned top-level follow-up dependency versions: `requirements-followup.txt` (not a complete transitive or OS lock)
- Model revisions and dataset fingerprints: `results/followup/provenance.json`
- 원본 진단: `tools/analyze_result_variance.py`
- 후속 실행: `tools/run_collapse_followup.py`
- 논문 표 생성: `tools/build_paper_followup_report.py`
- 원시 run 결과: `results/followup/collapse_followup_v2_*/**/final_metrics.json`, `predictions.csv`
