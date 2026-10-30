# 코드 보존 범위

Isaac Lab 작업본의 `src/`, `scripts/`, `pyproject.toml`을 복사했습니다.
기존 단계별 소스는 보존했습니다. 업로드용 변경은 다음에 한정됩니다.

- `scripts/evaluate_baseline.py` 추가: 외부 적분 평가 코드를 재사용하며 저장소 상대 경로의
  phase14 모델/설정을 읽습니다. 과거 결과 폴더·검증 결과·reference pt가 없어도 재생합니다.
  선택적인 `--reference`는 관측/action 검증 자료이며 모델 파일로 취급하지 않습니다.
- `scripts/build_asset.py`: 제외된 과거 evidence JSON 대신 동일한 task 물성 상수를 읽습니다.
- README, 기체 설명, manifest 및 .gitignore 추가.

원본 서버 프로젝트와 기존 MuJoCo 브랜치는 유지됩니다.
실험 보고서, CSV, PNG, 이벤트, reference observation pt, 압축파일,
다른 체크포인트는 포함하지 않습니다. `resources`는 기체 모델 원본입니다.
