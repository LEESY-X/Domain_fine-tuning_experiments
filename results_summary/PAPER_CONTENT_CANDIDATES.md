# 논문 삽입 후보 전체 체크리스트

이 문서는 현재 확보된 결과를 논문의 어느 부분에 사용할 수 있는지, 어떤 표·그림·문장을 추가하면 좋은지, 어떤 주장은 추가실험 없이는 쓰면 안 되는지를 정리한 편집 체크리스트다.

표시 기준:

- **즉시 사용 가능**: 현재 저장된 표·코드·run-level 증거로 뒷받침 가능
- **조건부 사용**: 한계 문장을 함께 써야 함
- **추가실험 필요**: 현재 결과만으로는 주장 불가

## 1. 논문의 중심 메시지 후보

| 우선순위 | 메시지 | 상태 | 권장 위치 |
|---:|---|---|---|
| 1 | 낮은 seed 표준편차는 항상 좋은 안정성이 아니며 constant-class collapse를 가릴 수 있다 | 즉시 사용 가능 | Abstract, Results, Discussion |
| 2 | 안정성 평가는 Macro-F1 SD와 함께 예측 class coverage·최빈 예측률·entropy를 보고해야 한다 | 즉시 사용 가능 | Method, Discussion, Checklist/Recommendation |
| 3 | Full fine-tuning은 22개 조건 중 19개에서 최고 Macro-F1을 보여 가장 강한 일반 기준선이었다 | 즉시 사용 가능 | Abstract, Main Results |
| 4 | PEFT의 성능은 방법 자체보다 task/model/optimization 조건에 크게 의존했다 | 조건부 사용 | Discussion |
| 5 | 문제 네 조건에서 결합 개선안은 20/20 paired seeds의 Macro-F1을 높이고 붕괴를 0/20으로 줄였다 | 즉시 사용 가능 | Follow-up Results |
| 6 | 붕괴 완화에는 baseline 대비 약 2.32–2.56배의 학습시간 비용이 동반됐다 | 즉시 사용 가능 | Results, Practical Trade-off |
| 7 | Parameter efficiency, wall-clock efficiency, predictive performance와 non-degenerate stability는 서로 다른 선택 기준이다 | 즉시 사용 가능 | Discussion, Conclusion |

## 2. 제목 후보

- `When Zero Variance Means Collapse: Auditing Stability in Parameter-Efficient Fine-Tuning`
- `Beyond Macro-F1 Standard Deviation: Detecting Degenerate Stability in PEFT`
- `Performance, Efficiency, and Degenerate Stability Across Full and Parameter-Efficient Fine-Tuning`
- `Seed Stability or Prediction Collapse? A Multi-Task Audit of Fine-Tuning Strategies`

제목에 “solving”이나 “eliminating”처럼 일반적인 해결을 암시하는 표현은 피한다. 후속 범위는 네 조건이며 remedy는 결합 설정이다.

## 3. Abstract에 들어갈 요소

### 배경 한 문장

- PEFT 연구는 평균 성능과 parameter efficiency를 주로 비교하지만, 낮은 seed variance가 실제 robustness인지 퇴화된 동일 예측인지 구분하지 않는 경우가 있다.

### 설계 한 문장

- 3개 base model, 13개 task, 22개 task/model 조건, 5개 fine-tuning 방법과 5개 seed로 구성된 550-run aggregate를 분석했다.

### 핵심 결과 후보

- Full fine-tuning이 22개 조건 중 19개에서 최고 Macro-F1을 기록했다.
- 원본 110개 method 집계 행 중 4개 exact-zero SD 행이 constant-class prediction의 analytical fingerprint와 일치했다.
- 후속 100-run 진단에서 Pilot baseline 20/20 collapse가 재현됐다.
- Full-data paired 비교에서 결합 개선 설정은 네 조건 모두 평균 Macro-F1을 높였고 remedy 20/20 run에서 collapse가 없었다.
- n=5 exact sign-flip 검정의 최소 양측 p-value가 0.0625이므로 효과 크기와 방향 일관성을 보고하되 conventional significance는 주장하지 않았다.

### 결론 한 문장

- Fine-tuning stability 평가는 표준편차만이 아니라 예측 분포와 class coverage를 함께 포함해야 한다.

## 4. Introduction에 추가하면 좋은 논리 흐름

1. Full FT와 PEFT를 비교할 때 성능·파라미터·시간이 각각 다른 목적함수임을 제시한다.
2. 여러 seed의 평균±SD가 일반적이지만 SD가 낮다는 사실만으로 올바른 수렴을 보장하지 않는다는 문제를 제기한다.
3. 모든 seed가 같은 다수 class만 예측하면 F1이 동일해져 SD가 0이 되는 반례를 수식으로 제시한다.
4. 연구 질문을 다음처럼 명시한다.
   - RQ1: Full FT와 네 PEFT 방법의 성능·시간·parameter efficiency는 task/model별로 어떻게 다른가?
   - RQ2: 낮은 seed variance 중 실제 안정성과 prediction collapse를 어떻게 구분할 수 있는가?
   - RQ3: 붕괴된 PEFT 조건은 학습시간, learning rate, class weighting을 조정했을 때 회복되는가?
5. 기여를 “새 universal PEFT 방법”이 아니라 “비교 실험 + 안정성 audit + collapse-aware reporting protocol”로 정의한다.

## 5. Related Work에서 다룰 항목

최종 원고에서는 각 항목의 원 논문과 공식 구현을 실제 참고문헌으로 확인해 인용한다.

- Full model fine-tuning과 transfer learning
- LoRA의 low-rank update와 parameter efficiency
- Bottleneck Adapter의 구조와 삽입 위치
- IA3의 activation scaling 방식
- BitFit의 bias-only fine-tuning
- BERTweet, RoBERTa, KLUE RoBERTa의 사전학습 domain 차이
- Imbalanced classification에서 weighted cross-entropy
- Macro-F1의 class-balanced 해석과 accuracy와의 차이
- Deep learning의 random-seed variability와 reproducibility
- Majority-class/constant-prediction collapse 및 degenerate solutions
- Exact randomization/sign-flip test와 small-n inference

## 6. Methods 본문에 반드시 들어갈 내용

### 실험 단위

- 독립 run 단위는 `task × model × method × seed`
- 원본 총 550 runs와 Study별 25/450/75 계산
- 후속 100 runs의 Pilot 60/Full 40 구분

### 재현 설정

- seed 5개와 data seed 42
- task별 split 정책과 split 크기
- max length 128, batch 16, eval batch 32, accumulation 4
- AdamW torch, linear scheduler, weight decay 0.01, warmup 0.06
- epoch별 evaluation/save, validation Macro-F1 best checkpoint, early stopping patience 2
- method별 learning rate와 PEFT architecture 설정
- CUDA 원본과 Apple MPS 후속 환경을 분리 표기

### Collapse-aware 평가 정의

- `predicted_class_count == 1`을 constant-class collapse로 정의
- majority-prediction rate 0.98 이상을 near-collapse flag로 정의
- predicted-class coverage와 normalized entropy를 보조 지표로 기록
- F1 SD만으로 stability winner를 정하지 않는 분석 원칙

### 통계

- 5 seeds 평균과 sample SD
- 동일 seed paired differences
- paired t interval은 uncertainty summary
- exact two-sided sign-flip test를 inferential check로 사용
- multiplicity adjustment 미적용 및 exploratory scope 명시

## 7. Main Results에 권장하는 표

### Table A. 전체 방법 비교

현재 자료: `results_summary/key_insights.md`

| Method | 평균 Macro-F1 | 성능 1위 조건 |
|---|---:|---:|
| Full FT | 0.7588 | 19/22 |
| IA3 | 0.6702 | 2/22 |
| LoRA | 0.6537 | 0/22 |
| Adapter | 0.6506 | 1/22 |
| BitFit | 0.6443 | 0/22 |

표 하단에는 task/model 조건에 동일 가중한 macro average인지 명시한다. dataset sample 수로 가중한 평균으로 오해되지 않게 한다.

### Table B. PEFT가 Full FT를 앞선 조건

- `tweet_hate` + RoBERTa: IA3, Macro-F1 0.5727
- `tweet_hate` + BERTweet: IA3, Macro-F1 0.5349
- `news_ynat` + KLUE RoBERTa: Adapter, Macro-F1 0.8702

이 세 조건은 “PEFT가 항상 열세”라는 단순 결론을 피하게 해준다. 다만 winner 차이의 CI나 paired 검정은 원본 run-level data가 없어 현재 계산할 수 없다.

### Table C. Exact-zero SD 진단

현재 자료: `paper_followup/table_1_original_zero_sd.csv`

- 네 조건의 accuracy, precision, recall, F1과 analytical constant-class 예상값을 함께 제시
- `f1_sd=0`을 “best stability”가 아니라 “degenerate stability”로 재분류

### Table D. Pilot ablation

현재 자료: `paper_followup/table_2_pilot_ablation.csv`

- Finance/RoBERTa/Adapter에서 6 variants 비교
- 단독/부분 변경 다섯 조건은 모두 5/5 collapse
- higher LR + weighting + longer schedule만 0/5 collapse, F1 `0.8040 ± 0.0063`

이 표는 요소별 주효과를 증명하지 않고 결합안 선별 과정을 보여주는 용도로 쓴다.

### Table E. Full-data paired follow-up

현재 자료: `paper_followup/table_3_full_comparison.csv`

- baseline/remedy mean±SD
- paired delta와 95% CI
- collapse counts
- majority-prediction rate
- improved seeds, full class coverage
- train-time multiplier
- exact p-value

이 표가 후속 실험의 주 결과표다.

## 8. 권장 그림

| 우선순위 | 그림 | 전달할 주장 | 현재 상태 |
|---:|---|---|---|
| 1 | 네 조건 baseline vs remedy grouped bar | 네 조건 모두 remedy 평균 F1이 높음 | HTML 보고서에 있음; 논문용 vector/PDF export 필요 |
| 2 | Seed별 paired slope plot | 20/20 paired comparisons가 같은 방향임 | Appendix seed CSV로 즉시 생성 가능 |
| 3 | F1 SD vs majority-prediction rate scatter | 낮은 SD가 collapse와 결합될 수 있음 | 후속 100 runs로 생성 가능 |
| 4 | Predicted-class coverage 또는 entropy distribution | 정상 안정성과 퇴화 안정성의 분리 | 후속 predictions로 생성 가능 |
| 5 | Performance vs trainable ratio Pareto plot | 성능과 parameter efficiency trade-off | 원본 110행 aggregate로 생성 가능 |
| 6 | Performance vs wall-clock time plot | parameter efficiency와 시간 효율이 다름 | 원본 aggregate로 가능하나 cross-study 절대시간 비교는 금지 |
| 7 | Task/model × method Macro-F1 heatmap | method ranking의 조건 의존성 | 원본 110행 aggregate로 생성 가능 |
| 8 | Collapse detection flow diagram | SD→fingerprint→prediction diagnostics→paired rerun 절차 | 코드와 정의로 제작 가능 |

그림 2–4는 후속 연구의 독창적인 진단 메시지를 가장 잘 보여준다. 평균 bar chart만 제시하면 seed 방향성과 collapse 메커니즘이 가려질 수 있다.

## 9. Results 해석 문장 후보

### 즉시 사용 가능한 문장

- “Full fine-tuning achieved the highest Macro-F1 in 19 of 22 task–model conditions, while IA3 and Adapter were competitive in a small number of task-specific settings.”
- “All four exact-zero standard-deviation rows matched the analytical metric fingerprint of constant-class prediction at full stored precision.”
- “All 20 limited-data follow-up baseline runs directly exhibited constant-class collapse; this was not an identical replay of the historical executions.”
- “In the full-data follow-up, the combined remedy improved Macro-F1 for every paired seed and reduced constant-class collapse to zero in all 20 remedy runs.”
- “The remedy increased mean training time by 2.32–2.56×, showing that collapse mitigation introduced a non-trivial compute trade-off.”

### 반드시 한계와 함께 쓸 문장

- “The combined remedy was consistently better across the tested seeds.” 뒤에 n=5와 exact p=0.0625를 명시한다.
- “The zero-SD rows are consistent with collapse.” 원본 run-level predictions가 없으므로 원본에 대해서는 `proved from predictions`가 아니라 `matched the aggregate fingerprint`라고 쓴다.
- “BERTweet LoRA recovered on the full dataset.” Pilot의 일부 seed 불안정성과 Full-data에서의 회복을 구분한다.

## 10. Discussion에 들어갈 인사이트

### Stability metric의 재정의

- 낮은 SD는 평균 성능이 의미 있는 해에 도달했을 때만 desirable stability다.
- 안정성 보고는 평균/SD뿐 아니라 collapse rate, class coverage, majority rate, entropy를 포함해야 한다.
- stability winner 선정 전에 minimum performance 또는 non-collapse gate를 적용하는 방안을 제안할 수 있다.

### PEFT optimization sensitivity

- 적은 trainable parameter가 더 빠른 optimization이나 더 안정적인 optimization을 자동 보장하지 않는다.
- 특히 짧은 2-epoch schedule과 class imbalance가 일부 PEFT 조건에서 다수 class 고착을 만들 가능성이 있다.
- 단, 현재 결합 실험만으로 under-training과 imbalance의 개별 기여를 분리할 수 없다.

### Full FT와 PEFT의 실제 선택 기준

- 최고 성능이 목적이면 Full FT가 강한 default다.
- parameter budget이 핵심이면 IA3가 모든 22개 조건에서 최소 trainable ratio였다.
- 특정 task에서는 IA3/Adapter가 Full FT를 앞섰으므로 task-aware selection이 필요하다.
- wall-clock time과 parameter ratio는 별도 축으로 보고해야 한다.

### Pilot과 Full-data의 차이

- BERTweet-LoRA emotion remedy의 Pilot SD는 컸지만 Full-data에서는 `0.7840 ± 0.0037`로 안정화됐다.
- Pilot은 빠른 조건 선별에는 유용하지만 최종 효과 추정치로 사용하면 안 된다는 사례가 된다.

## 11. Limitations와 Threats to Validity

### Internal validity

- Full-data remedy가 세 요소를 동시에 변경해 component-level causality를 분리하지 못함
- 원본 550회 실행 당시 optimizer/scheduler의 per-run 기록 부재
- 후속 실행 source snapshot과 현재 source의 hash 차이
- checkpoint 삭제로 exact weight-level 재검증 불가

### Statistical conclusion validity

- seed 5개로 검정력이 낮고 exact 양측 p-value가 0.0625 아래로 내려갈 수 없음
- 네 조건 p-value에 multiple-comparison correction 미적용
- paired t interval은 small-n normality assumption에 민감

### Construct validity

- collapse 정의 `predicted_class_count=1`은 명확하지만 near-collapse 0.98 threshold는 연구자 선택임
- Macro-F1 하나로 calibration, per-class harm, robustness를 모두 설명할 수 없음
- trainable parameter ratio는 memory footprint나 inference cost와 동일하지 않음

### External validity

- 후속 실험이 원본 문제 네 조건에만 한정됨
- 모델은 RoBERTa/BERTweet 중심이고 generative LLM에 직접 일반화할 수 없음
- 영어/한국어 text classification 결과를 다른 domain·language·sequence length에 일반화할 수 없음

### Reproducibility validity

- 원본 550회 run-level 결과 부재
- dataset revisions와 KLUE model revision 미고정
- 원본 CUDA와 후속 Apple MPS 환경 차이
- dataset license와 외부 source availability에 의존

## 12. 추가실험 우선순위

### 최우선: 결론을 강하게 만드는 실험

1. **전체 데이터 2×2×2 factorial ablation**
   - LR: original/higher
   - Weighting: none/inverse-sqrt
   - Epoch: 2/최대 5
   - 네 조건에 같은 5 seeds를 적용해 main effect와 interaction을 분리

2. **Seed 수 확대**
   - 최소 10개 이상 seed를 사전 정의
   - exact/randomization test의 해상도를 높이고 CI 안정성을 확인

3. **원본 CUDA 환경 재현**
   - 동일 model commit, dataset fingerprint, code tag로 핵심 4개 조건 재실행
   - MPS에서만 발생한 결과인지 분리

4. **Checkpoint별 collapse trajectory**
   - validation predicted-class count, majority rate, entropy, per-class recall을 epoch/checkpoint별 저장
   - 붕괴가 초기화 직후 고착되는지, 학습 중 발생하는지 확인

### 두 번째 우선순위: 메커니즘과 일반화

5. Class weight 방식 비교: inverse frequency, inverse sqrt, focal loss, balanced sampler
6. LR sweep: method별 log-scale grid와 warmup sensitivity
7. Early-stopping 기준 비교: Macro-F1, loss, collapse-aware composite metric
8. Data-size learning curve: 1k, 2k, 4k, full data
9. Class imbalance severity를 통제한 synthetic/downsample experiment
10. 다른 base model과 task에서 collapse-aware protocol replication
11. Per-class precision/recall과 confusion matrix 비교
12. Calibration: ECE/Brier score 및 confidence distribution

### 운영·효율 확장

13. Peak GPU/MPS memory와 energy 사용량 기록
14. End-to-end time을 model loading, training, evaluation으로 분해
15. Checkpoint 크기, optimizer-state 크기, inference latency 비교

## 13. Appendix/Supplement에 넣을 항목

- 전체 13개 dataset source·split·label mapping 표
- 22개 task/model × 5 methods 전체 결과표
- method별 architecture와 trainable parameter 수
- seed별 20개 paired 결과
- 각 condition의 prediction count·coverage·entropy
- Pilot 6-variant ablation 전체 표
- exact-zero fingerprint 유도식
- class weight 계산식
- paired CI와 sign-flip test 정의
- model commit과 dataset fingerprint
- software/hardware environment
- 실행 명령과 파일 구조
- 실패·resume·stale-result 처리 방식
- checksum 또는 release manifest

## 14. Artifact Availability 문단에 포함할 내용

- 공개되는 것: source, config, notebooks, original aggregate, follow-up per-run metrics/predictions/history, derived tables
- 공개되지 않는 것: dataset cache, raw dataset, checkpoints, original 550-run per-example predictions
- 재생성 명령과 exact dependency file 위치
- model commit과 dataset fingerprint 위치
- license와 data redistribution은 원천 dataset 규정 적용
- 익명 심사라면 repository URL과 author-identifying 경로/output가 제거됐는지 재확인

## 15. Ethics와 데이터 거버넌스

- hate/offensive speech dataset에는 유해 언어와 집단 관련 내용이 포함될 수 있음을 명시
- dataset별 consent, license, intended use와 demographic annotation 한계를 원 출처 기준으로 검토
- K-MHaS와 measuring-hate-speech의 label 이진화가 원래 label 의미를 단순화한다는 점 명시
- class weighting이 minority class recall을 높일 수 있지만 false positive 분포도 함께 확인해야 함
- 모델 결과를 실제 moderation 결정으로 직접 사용하는 연구가 아님을 구분

## 16. 주장 금지 목록

- “표준편차 0이므로 가장 안정적이다.”
- “원본 predictions로 네 붕괴를 직접 확인했다.”
- “Class weighting 하나가 붕괴를 해결했다.”
- “개선안이 통계적으로 유의하다, p<0.05.”
- “모든 PEFT 방법과 모든 task에 일반화된다.”
- “Trainable parameter가 적으므로 wall-clock, memory, energy도 항상 적다.”
- “MPS와 CUDA 결과는 직접 동일하게 비교할 수 있다.”
- “Pilot 결과가 Full-data 효과 크기를 정확히 추정한다.”
- “현재 repository만으로 원본 550회 실행을 bitwise 재현할 수 있다.”

## 17. 제출 전 편집 체크리스트

- [ ] Abstract 수치가 최종 CSV와 일치하는가?
- [ ] 원본 550회 결과와 후속 100회 결과가 명확히 구분되는가?
- [ ] `mean ± sample SD`, seed 수와 comparison unit을 모든 표 caption에 적었는가?
- [ ] Collapse와 near-collapse 정의가 threshold와 함께 적혀 있는가?
- [ ] Exact p=0.0625를 p<0.05로 잘못 해석하지 않았는가?
- [ ] Paired CI와 exact test의 역할을 구분했는가?
- [ ] Remedy가 세 요소의 결합임을 적었는가?
- [ ] 원본 CUDA/후속 MPS 차이를 limitation에 적었는가?
- [ ] Dataset·model·method 원 논문을 모두 인용했는가?
- [ ] Dataset license와 redistribution 조건을 확인했는가?
- [ ] Release commit/tag, checksum, DOI 또는 archive URL을 기록했는가?
- [ ] 익명 심사본에서 author·machine-local 정보가 제거됐는가?
- [ ] 논문 표와 appendix 표가 동일 generator에서 만들어졌는가?
- [ ] 그림은 vector PDF/SVG 또는 저널 규격 해상도로 export했는가?
- [ ] Code Availability와 Data Availability 문단이 실제 공개 범위와 일치하는가?
