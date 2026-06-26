# Sensitive Information Check

본 점검은 GitHub 공개 전 민감정보 및 대용량 파일 위험을 확인하기 위해 수행하였다. 원본 파일은 수정하지 않았고, 파일 내용은 읽기 전용으로만 확인하였다.

## 1. 점검 방식

다음 파일 유형을 대상으로 키워드 검색을 수행하였다.

- `.ipynb`
- `.py`
- `.json`
- `.yaml`, `.yml`
- `.env`
- `.txt`
- `.md`
- `.csv`

검색 키워드:

`api_key`, `apikey`, `secret`, `token`, `password`, `passwd`, `authorization`, `bearer`, `hf_`, `sk-`, `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`, `ssh`, `private`, `credential`, `auth`

## 2. 민감정보 점검표

| 점검 항목 | 발견 여부 | 위치 | 조치 필요 여부 | 비고 |
|---|---|---|---|---|
| API key | 발견 안 됨 | 검색 대상 파일 | 아니오 | `api_key`, `apikey` 직접 노출 없음 |
| Hugging Face token | 발견 안 됨 | 검색 대상 파일 | 아니오 | `hf_` 형식 token 노출 없음 |
| OpenAI API key | 발견 안 됨 | 검색 대상 파일 | 아니오 | `sk-`, `OPENAI_API_KEY` 노출 없음 |
| wandb key | 발견 안 됨 | 검색 대상 파일 | 아니오 | `WANDB_API_KEY` 노출 없음 |
| `.env` 파일 | 발견 안 됨 | repository tree | 아니오 | `.env` 파일 미확인 |
| 개인 이메일 | 발견 안 됨 | 검색 대상 파일 | 아니오 | email pattern 미확인 |
| 로컬 절대경로 | 발견됨 | `README.md`, `NOTION_EXPERIMENT_GUIDE.md`, `START_HERE_NEXT_SESSION.md`, `CONTINUE_FROM_HERE_2026-06-26.md` | 예 | `C:\Users\...` 형태 경로가 일부 문서에 존재 |
| 개인 이름/계정명 | 발견됨 | 로컬 절대경로 포함 문서 | 예 | `C:\Users\LEESY\...` 형태 계정명이 노출됨 |
| 비공개 데이터셋 | 확인 필요 | `cache/`, `src/suite.py` | 확인 필요 | 원천 데이터셋 라이선스 및 cache 재배포 가능 여부 확인 필요 |
| 대용량 checkpoint | 발견됨 | `results/**/checkpoints/` | 예 | 100MB 이상 파일 444개, checkpoint-like 파일 3330개 확인 |
| 서버 주소 또는 SSH 정보 | 발견 안 됨 | 검색 대상 파일 | 아니오 | `ssh` credential 형태 미확인 |
| 인증 정보 | 발견 안 됨 | 검색 대상 파일 | 아니오 | 검색 키워드 기준 credential 직접 노출 없음 |

## 3. 확인된 일반 keyword match

다음 항목은 민감정보가 아니라 코드 구조상 자연스럽게 나타난 keyword match로 판단된다.

| Match 유형 | 예시 위치 | 판단 |
|---|---|---|
| tokenizer 관련 문자열 | `src/suite.py`, notebook 내부 source string, checkpoint tokenizer files | 민감정보 아님 |
| `auth` substring | 일부 일반 문자열 또는 코드 문맥 | 민감정보 직접 노출 아님 |
| `token` substring | `AutoTokenizer`, `tokenized`, `tokenizer_config.json` | 민감정보 아님 |

## 4. 대용량 파일 점검

| 항목 | 값 | 비고 |
|---|---:|---|
| 전체 파일 수 | 10569 | 읽기 전용 scan 기준 |
| 전체 폴더 크기 | 약 269.45 GB | 읽기 전용 scan 기준 |
| checkpoint-like 파일 수 | 3330 | `.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt` |
| 100MB 이상 파일 수 | 444 | GitHub 일반 업로드 부적합 |
| 최대 파일 예시 | 약 1029.40 MB | `results/study2/PAPER/product_reviews/.../optimizer.pt` |

## 5. 공개 전 권고

1. `results/**/checkpoints/`는 GitHub 일반 repository에 직접 업로드하지 않는다.
2. 100MB 이상 파일은 Git LFS 또는 외부 artifact storage를 사용한다.
3. `cache/`는 원천 데이터셋 라이선스를 확인하기 전에는 공개 repository에서 제외한다.
4. 기존 문서의 로컬 절대경로와 계정명은 공개 전 제거하거나 상대경로로 바꾼다.
5. 원본 실험 파일을 수정하지 않아야 한다면 공개용 별도 branch 또는 export bundle에서 문서만 정리한다.

