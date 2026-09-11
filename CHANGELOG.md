# 변경 내용

## 0.2.0 — 2026-09-11

- PDF/HTML 보고서와 보고서 생성 script 제거
- 초보자 매뉴얼, 이전 코드 문제·수정 근거, 수식-코드 대응 문서 추가
- 0번 sample 이중 적분을 수정하고 회귀 테스트 추가
- InEKF right perturbation/correction 설명을 코드와 맞추고 Jacobian 테스트 추가
- SO(3) geodesic attitude RMSE/final error 추가
- 실행 설정, bias, 초기 상태, update 횟수를 `run_manifest.json`에 저장
- 재현 불가능한 historical 수치·그래프 제거
- 결과 근거를 `results/` 디렉터리로 분리

## 0.1.0 — 2026-09-05

- EKF, UKF, PF, ESKF, InEKF를 하나의 Python API로 실행하도록 정리
- 6축 IMU 입력과 6차원 pose 출력 형식 통일
- EuRoC IMU-only와 위치 update 실험 설정 추가
- 회전 운동을 세 단계로 바꾼 합성 실험 추가
- CF231 Small TCN 학습과 InEKF velocity update 추가
- 학습에서 제외한 Run 5의 평가 결과 저장
- 결과 CSV·JSON, 그래프 생성 코드와 사용 설명 추가
- EuRoC에서 발산한 ESKF와 Pohang의 update/reference 중복 사용 문제 기록
