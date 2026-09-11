# 코드 출처와 변경 범위

## 1. 지난 학기 benchmark

- 저장소: <https://github.com/INHA-Artemis/State_Estimation_Benchmark>
- 확인 commit: `2b9199f993703a98b811fccd7631a06b87763ff0`

지난 학기에는 여러 외부 필터 구현을 데이터셋별 script로 실행하고 그래프·영상을 생성했습니다. 현재 코드와 파일 구조가 다르므로 지난 학기 수치를 현재 필터의 성능표에 직접 합치지 않습니다.

## 2. 이번 작업의 출발 코드

- 저장소: <https://github.com/andyjaehun/State_Estimation>
- 기준 commit: `0967c4737f803aad897e1ba6e16211fb26288821`
- commit 일시: 2026-08-13 14:34:16 +09:00

가져온 부분:

- EKF, UKF, PF, ESKF, InEKF 필터 class
- EuRoC, i2Nav, Pohang, synthetic loader의 기본 구조
- filter registry, 설정, metric, visualization
- `runners/run_filter.py`, `run_all.py`의 기본 구조

## 3. 이 저장소에서 추가한 부분

- 설치 가능한 `state_estimation` API와 command-line entry point
- EuRoC 연속 IMU-only 설정
- 초기 sample 이중 적분 수정과 회귀 테스트
- right perturbation InEKF 설명·Jacobian·correction 테스트
- 3D motion-regime generator/runner
- CF231 loader, 초기 고정 bias, gyro intrinsic fit
- CF231 Run 5 held-out Small TCN 학습과 InEKF velocity update
- SO(3) geodesic attitude metric
- 실행별 `run_manifest.json`
- 데이터셋 준비, 코드 흐름, 수식, 결과 해석 문서

## 4. 외부 참고 구현

InEKF propagation convention은 `ghaggin/invariant-ekf`의 `SE_2(3)` LIEKF 구현과 비교했습니다. 0722 validation에서 upstream commit `35855644064e4cfdfb67b83ff87b90af7f66fbf2`의 C++ propagation과 CF231 Run 5에 동일 초기 상태·bias·IMU를 넣었을 때 최대 차이는 다음과 같았습니다.

```text
SO(3)       1.41e-13 deg
velocity    5.69e-12 m/s
position    5.88e-10 m
```

이 비교는 **measurement update가 없는 nominal propagation이 reference와 같다**는 것을 확인합니다. Kalman covariance/update 전체가 외부 구현과 동일하다는 증명은 아닙니다.

## 5. 결과 출처

- `results/data/euroc_results.csv`: `config/euroc.yaml`, `euroc_imu_only.yaml`로 전체 재실행
- `results/data/motion_regime_metrics.csv`: `config/motion_regimes.yaml`로 생성
- `results/data/cf231_run5_summary.json`: `runners/run_cf231_small_tcn.py`의 training/CV/test 출력

원 설정과 output이 없는 historical CSV는 현재 결과에서 제외했습니다.
